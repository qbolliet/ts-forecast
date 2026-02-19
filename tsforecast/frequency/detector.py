"""Frequency detection utilities for time series data.

This module provides the FrequencyDetector class to detect and validate frequencies
in time series and panel data, with primary reliance on pandas.infer_freq.
"""
# Importation des modules
# Module de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from pandas.tseries.frequencies import to_offset

# Import des utilitaires de fréquence
from ..utils.frequency import to_literal, get_frequency_order, normalize_frequency, FrequencyType, UserFrequencyType
from ..panel.utils import normalize_entity_key
from ..utils.parse.utils import parse_frequency

# Classe de détection de la fréquence d'une série temporelle
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
    # Initialisation
    def __init__(self, min_observations: int = 2):
        """Initialize the FrequencyDetector.

        Args:
            min_observations: Minimum number of observations required to detect frequency
        """
        self.min_observations = min_observations

    # Méthode de détection d'une série temporelle simple
    def detect_time_series_frequency(self, series: pd.Series, literal: bool=False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect the frequency of a single time series with DatetimeIndex.

        This method primarily uses pandas.infer_freq with extensions for frequencies
        not natively supported (like quarterly). It automatically drops NaN values
        before detection.

        Args:
            series: Time series data with datetime index
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Detected frequency as user-friendly string, or None if detection fails

        Raises:
            ValueError: If series has insufficient non-null observations or invalid index

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, np.nan, 4, 5], index=dates)
            >>> detector = FrequencyDetector()
            >>> detector.detect_time_series_frequency(series)
            'daily'
        """
        # Suppression des valeurs manquantes pour la détection
        clean_series = series.dropna()

        # Vérification que le nombre d'observations dans le série est supérieur au minimum requis
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
            # Normalisation
            inferred_freq = normalize_frequency(frequency=inferred_freq, return_format='base')
            if inferred_freq and literal:
                # Conversion vers le format littéral
                return to_literal(inferred_freq)
            elif inferred_freq:
                return inferred_freq
        except Exception:
            # En cas d'erreur avec infer_freq, continuer avec la détection manuelle
            pass

        # Extension pour les fréquences non supportées par infer_freq
        extended_freq = self._extend_infer_freq(time_index, literal)
        # Normalisation
        extended_freq = normalize_frequency(frequency=extended_freq, return_format='base')
        if extended_freq and literal:
            # Conversion vers le format littéral
            return to_literal(extended_freq)
        elif extended_freq:
            return extended_freq

        return None

    # Méthode auxiliaire pour extraire la série temporelles (en dernière dimension par convention) d'un multi-index
    def _extract_time_series_from_multiindex(self, series: pd.Series) -> pd.Series:
        """Extract a simple time series from a series with MultiIndex.

        Args:
            series: Series with MultiIndex where the last level is the date

        Returns:
            Simple time series with only the time index
        """
        # Extraction de l'index temporel (dernier niveau)
        time_index = series.index.get_level_values(-1)
        # Création d'une série temporelle simple avec l'index temporel
        return pd.Series(series.values, index=time_index)

    # Méthode auxiliaire d'extraction de la fréquence d'une colonne
    def _detect_column_frequency(self, series: pd.Series, literal: bool) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect the frequency of a column (with automatic MultiIndex handling).

        Args:
            series: Series for which to detect the frequency
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Detected frequency or None if failed/insufficient observations
        """
        try:
            # Si la série a un MultiIndex, extraire la série temporelle simple
            if isinstance(series.index, pd.MultiIndex):
                series = self._extract_time_series_from_multiindex(series)
            return self.detect_time_series_frequency(series, literal)
        except ValueError:
            # Pas assez d'observations, retourner None
            return None

    # Méthode de détection de la fréquence d'une série (simple ou avec MultiIndex)
    def detect_frequency(self, series: pd.Series, literal: bool=False) -> Optional[Union[FrequencyType, UserFrequencyType, Dict[tuple, Union[FrequencyType, UserFrequencyType]]]]:
        """Detect the frequency of a series (simple or with MultiIndex for panel data).

        This method handles both simple time series and panel data with MultiIndex.
        For MultiIndex, it groups by all levels except the last (assumed to be the date)
        and detects frequency for each panel group.

        Args:
            series: Time series data with DatetimeIndex or MultiIndex
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            - For simple series: Detected frequency as string, or None if detection fails
            - For MultiIndex series: Dictionary mapping panel_id to frequencies

        Raises:
            ValueError: If series has insufficient non-null observations or invalid index

        Examples:
            >>> import pandas as pd
            >>> # Simple time series
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
            >>> detector = FrequencyDetector()
            >>> detector.detect_frequency(series)
            'D'

            >>> # Panel data with MultiIndex
            >>> idx = pd.MultiIndex.from_arrays([
            ...     ['A', 'A', 'A', 'B', 'B', 'B'],
            ...     pd.date_range('2023-01-01', periods=3, freq='D').tolist() * 2
            ... ], names=['panel_id', 'date'])
            >>> series = pd.Series([1, 2, 3, 4, 5, 6], index=idx)
            >>> detector.detect_frequency(series)
            {'A': 'D', 'B': 'D'}
        """
        # Cas d'une série temporelle simple
        if not isinstance(series.index, pd.MultiIndex):
            return self.detect_time_series_frequency(series, literal)

        # Cas d'une série avec MultiIndex (panel data)
        n_levels = series.index.nlevels

        # Vérification qu'il y a au moins 2 niveaux (panel_id + date)
        if n_levels < 2:
            raise ValueError(
                f"MultiIndex must have at least 2 levels (panel_id and date), "
                f"but has only {n_levels}"
            )

        # Extraction des noms des niveaux de panel (tous sauf le dernier)
        panel_levels = list(range(n_levels - 1))
        frequency_map = {}

        # Groupby sur les niveaux de panel et détection pour chaque groupe
        for panel_values, group_series in series.groupby(level=panel_levels):
            # Création de l'identifiant du panel normalisé
            panel_id = normalize_entity_key(panel_values)

            # Extraction de la série temporelle simple et détection de la fréquence
            temp_series = self._extract_time_series_from_multiindex(group_series)
            freq = self._detect_column_frequency(temp_series, literal)

            if freq:
                frequency_map[panel_id] = freq

        # Retour du dictionnaire (None si vide)
        return frequency_map if frequency_map else None

    # Méthode auxiliaire de détection de la structure de panel
    def _detect_panel_structure(self, df: pd.DataFrame, panel_cols: Optional[List[str]]) -> Tuple[Optional[List[str]], bool]:
        """Detect the panel structure of the DataFrame (columns or index).

        Args:
            df: DataFrame to analyze
            panel_cols: List of specified panel columns (can be None)

        Returns:
            Tuple (panel_cols_final, panel_in_index) where:
                - panel_cols_final: List of detected panel columns (None if no panel)
                - panel_in_index: True if panel is in the index, False if in columns
        """
        # Initialisation de l'indicateur que le panel est en index
        panel_in_index = False
        # Initialisation des noms des colonnes de panel
        panel_cols_final = panel_cols

        # Auto-détection si panel_cols non spécifié et index MultiIndex
        if panel_cols is None and isinstance(df.index, pd.MultiIndex):
            if df.index.nlevels >= 2:
                # Extraction des noms des niveaux de panel (tous sauf le dernier)
                panel_cols_final = df.index.names[:-1]
                # Conversion en liste si nécessaire
                if not isinstance(panel_cols_final, list):
                    panel_cols_final = list(panel_cols_final)
                panel_in_index = True
        # Vérification si panel_cols spécifiés sont dans l'index
        elif panel_cols is not None and isinstance(df.index, pd.MultiIndex):
            if all(col in df.index.names for col in panel_cols):
                panel_cols_final = panel_cols
                panel_in_index = True

        return panel_cols_final, panel_in_index

    # Méthode de détection des fréquences d'un jeu de données de panel
    def _detect_panel_frequencies(self, df: pd.DataFrame, panel_cols: List[str], panel_in_index: bool,
                                  time_col: Optional[str], literal: bool) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a panel DataFrame.

        Args:
            df: DataFrame with panel structure
            panel_cols: List of panel columns
            panel_in_index: True if panel is in the index
            time_col: Name of the time column (or None if in the index)
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Dictionary mapping (panel_id, column) to frequencies
        """
        # Initialisation du dictionnaire des fréquences
        frequency_map = {}

        if panel_in_index:
            # Groupby par niveaux d'index
            for panel_values, group_df in df.groupby(level=panel_cols):
                # Création de l'identifiant du panel
                panel_id = normalize_entity_key(panel_values)

                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    if col != time_col:
                        # Extraction de la série temporelle simple depuis le MultiIndex
                        series_with_multiindex = group_df[col]
                        simple_series = self._extract_time_series_from_multiindex(series_with_multiindex)

                        # Détection de la fréquence
                        freq = self._detect_column_frequency(simple_series, literal)
                        if freq:
                            frequency_map[panel_id + (col,)] = freq
        else:
            # Groupby par colonnes
            for panel_values, group_df in df.groupby(panel_cols):
                # Création de l'identifiant du panel
                panel_id = normalize_entity_key(panel_values)

                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    if col not in panel_cols and col != time_col:
                        # Détection de la fréquence
                        freq = self._detect_column_frequency(group_df[col], literal)
                        if freq:
                            frequency_map[panel_id + (col,)] = freq

        return frequency_map

    # Méthode auxiliaire de détection des fréquences pour les jeux de données qui sont des séries temporelles
    def _detect_time_series_frequencies(self, df: pd.DataFrame, time_col: Optional[str],
                                   literal: bool) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a simple DataFrame (non-panel).

        Args:
            df: DataFrame
            time_col: Name of the time column (or None if in the index)
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Dictionary mapping column (or panel_id, column) to frequencies
        """
        # Initialisation du dictionnaire des fréquences
        frequency_map = {}

        # Traitement des séries temporelles
        for col in df.columns:
            if col != time_col:
                # Détection de la fréquence (peut retourner un dict si la colonne a un MultiIndex)
                freq_result = self.detect_frequency(df[col], literal)

                # Si le résultat est un dictionnaire (colonne avec MultiIndex)
                if isinstance(freq_result, dict):
                    # Fusion des résultats dans le frequency_map
                    for panel_id, freq in freq_result.items():
                        # Clé combinée (panel_id, col)
                        combined_key = (panel_id, col) if not isinstance(panel_id, tuple) else (*panel_id, col)
                        frequency_map[combined_key] = freq
                # Sinon, ajout direct de la fréquence
                elif freq_result:
                    frequency_map[col] = freq_result

        return frequency_map

    # Méthode de détection de la fréquence d'un jeu de données
    def detect_dataset_frequency(self,
                               df: pd.DataFrame,
                               time_col: Optional[str] = None,
                               panel_cols: Optional[List[str]] = None,
                               literal: bool = False) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for all series in a dataset.

        This method handles both simple DataFrames and panel data. Panel structure can be
        specified either via panel_cols or detected automatically from a MultiIndex.

        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column (if None, uses index)
            panel_cols: List of columns identifying panel dimensions. If None and index is
                MultiIndex with at least 2 levels, automatically extracts panel structure
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Dictionary mapping column names (or (panel_id, column) tuples) to frequencies

        Examples:
            >>> import pandas as pd
            >>> # Simple DataFrame
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> df = pd.DataFrame({'value1': [1, 2, 3, 4, 5], 'value2': [10, 20, 30, 40, 50]}, index=dates)
            >>> detector = FrequencyDetector()
            >>> freq_map = detector.detect_dataset_frequency(df)
            >>> freq_map
            {'value1': 'D', 'value2': 'D'}

            >>> # DataFrame with MultiIndex (automatic panel detection)
            >>> idx = pd.MultiIndex.from_arrays([
            ...     ['A', 'A', 'B', 'B'],
            ...     pd.date_range('2023-01-01', periods=2, freq='D').tolist() * 2
            ... ], names=['panel_id', 'date'])
            >>> df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)
            >>> freq_map = detector.detect_dataset_frequency(df)
            >>> freq_map
            {('A', 'value'): 'D', ('B', 'value'): 'D'}
        """
        # Préparation de l'index temporel si spécifié
        if time_col is not None and time_col in df.columns:
            df = df.set_index(time_col)

        # Détection de la structure panel (dans l'index ou les colonnes)
        panel_cols, panel_in_index = self._detect_panel_structure(df, panel_cols)

        # Détermination si les données sont en panel
        is_panel = panel_cols is not None and len(panel_cols) > 0

        # Détection des fréquences selon le type de structure
        if is_panel:
            return self._detect_panel_frequencies(df, panel_cols, panel_in_index, time_col, literal)
        else:
            return self._detect_time_series_frequencies(df, time_col, literal)

    # Méthode de validation de la consistence de la fréquence un jeu de données
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
        # Si le dictionnaire des fréquences de chaque colonne n'est pas spécifié, ne renvoie rien 
        if not frequency_map:
            return False, None

        # Détection des fréquences
        unique_frequencies = set(frequency_map.values())

        # Cas des fréquences uniques
        if len(unique_frequencies) == 1:
            return True, list(unique_frequencies)[0]

        # Dans le cas où les fréquences ne sont pas unique et que la validation est stricte
        if strict:
            return False, None

        # En mode non-strict, recherche de la fréquence la plus commune
        freq_counts = {}
        # Parcours des valeurs
        for freq in frequency_map.values():
            # Incrémentation de chaque fréquence
            freq_counts[freq] = freq_counts.get(freq, 0) + 1
        
        # Identification de la fréquence la plus commune
        most_common = max(freq_counts, key=freq_counts.get)

        return True, most_common

    # Méthode d'extension de la détection des fréquences au delà de ce que supporte infer_freq
    def _extend_infer_freq(self, time_index: pd.DatetimeIndex, literal: bool = False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Extend frequency detection for frequencies not supported by pandas.infer_freq.

        This method provides custom detection for frequencies like quarterly, business daily,
        semi-monthly, and sub-daily frequencies that pandas.infer_freq doesn't always detect
        reliably.

        Args:
            time_index: Sorted datetime index
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Extended frequency detection result in pandas format or None

        Notes:
            Returns pandas frequency codes ('D', 'B', 'W', 'SM', 'M', 'Q', 'Y', 'h', 'min', 's')
            to be compatible with FrequencyNormalizer.
        """
        # Ne retourne rien si le nombre d'observations n'est pas suffisant
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

        modal_diff = mode_diff[0]
        
        # Conversion en secondes pour une détection plus précise
        modal_seconds = modal_diff.total_seconds()
        
        # Détection des fréquences infra-journalières
        if modal_seconds < 86400:  # Moins d'un jour
            return self._detect_intraday_frequency(modal_seconds=modal_seconds, literal=literal)
        else:
            # Conversion en jours pour les fréquences >= journalières
            modal_days = modal_diff.days
            return self._detect_day_frequency(time_index=time_index, modal_days=modal_days, literal=literal)

    # Méthode auxiliaire de détection des basses fréquences en jours
    def _detect_day_frequency(self, time_index: pd.DatetimeIndex, modal_days: float, literal: bool = False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect day-level frequencies (daily, weekly, monthly, quarterly, annual).

        Args:
            time_index: Sorted datetime index
            modal_days: Modal time difference in days between consecutive observations
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Detected frequency or None if no match found
        """
        # Détection basée sur le nombre de jours modal
        if modal_days == 1:
            # Distinction entre jours calendaires et jours ouvrés
            return self._detect_daily_frequency(time_index=time_index, literal=literal)
        elif modal_days == 7:
            return to_literal('W') if literal else 'W'
        elif 13 <= modal_days <= 16:
            # Fréquence semi-mensuelle (environ 2 fois par mois)
            return self._detect_semi_monthly_frequency(time_index=time_index, literal=literal)
        elif 28 <= modal_days <= 31:
            return to_literal('M') if literal else 'M'
        elif 89 <= modal_days <= 92:
            return to_literal('Q') if literal else 'Q'
        elif 365 <= modal_days <= 366:
            return to_literal('Y') if literal else 'Y'

        return None

    # Méthode de détection des fréquences infrajournalières
    def _detect_intraday_frequency(self, modal_seconds: float, literal: bool = False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect intraday frequency (hourly, minute, second).

        Args:
            modal_seconds: Modal time difference in seconds
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            Pandas frequency code for intraday frequency or None

        Notes:
            Detects hourly ('h'), minute ('min'), and second ('s') frequencies.
        """
        # Tolérance de 5% pour gérer les petites variations
        tolerance = 0.05
        
        # Fréquence horaire
        if abs(modal_seconds - 3600) / 3600 < tolerance:
            return to_literal('h') if literal else 'h'
        
        # Fréquence par minute
        if abs(modal_seconds - 60) / 60 < tolerance:
            return to_literal('min') if literal else 'min'
        
        # Fréquence par seconde
        if abs(modal_seconds - 1) / 1 < tolerance:
            return to_literal('s') if literal else 's'
        
        # Sous-seconde (millisecondes, microsecondes, nanosecondes)
        if modal_seconds < 1:
            if modal_seconds >= 0.001:
                return to_literal('ms') if literal else 'ms'
            elif modal_seconds >= 0.000001:
                return to_literal('us') if literal else 'us'
            else:
                return to_literal('ns') if literal else 'ns'
        
        return None

    # Méthode auxiliaire de détection des fréquences journalières
    def _detect_daily_frequency(self, time_index: pd.DatetimeIndex, literal: bool = False) -> Union[FrequencyType, UserFrequencyType]:
        """Detect whether daily frequency is calendar daily or business daily.

        Args:
            time_index: Sorted datetime index
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            'D' for calendar daily or 'B' for business daily

        Notes:
            Checks if dates fall only on business days (Monday-Friday).
        """
        # Vérification des jours de la semaine (0=lundi, 6=dimanche)
        weekdays = time_index.dayofweek
        
        # Si tous les jours sont des jours ouvrés (0-4), c'est probablement business daily
        if all(day < 5 for day in weekdays):
            # Vérification supplémentaire: pas de week-ends dans la période
            date_range = pd.date_range(time_index.min(), time_index.max(), freq='D')
            weekend_dates = date_range[date_range.dayofweek >= 5]
            
            # Si la période couvre des week-ends mais qu'ils sont absents, c'est business daily
            if len(weekend_dates) > 0:
                return to_literal('B') if literal else 'B'
        
        return 'D'

    # Méthode auxiliaire de détection des fréquences semi-mensuelles
    def _detect_semi_monthly_frequency(self, time_index: pd.DatetimeIndex, literal: bool=False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect semi-monthly frequency (twice per month).

        Args:
            time_index: Sorted datetime index
            literal: Boolean indicating whether to return explicit literal frequency expression

        Returns:
            'SM' if semi-monthly frequency is detected, None otherwise

        Notes:
            Checks if dates correspond to bi-monthly occurrences
            (typically 1st and 15th of the month, or beginning and middle of the month).
        """    
        # Extraction des jours du mois
        days = time_index.day
        
        # Vérification si les dates sont regroupées autour de 2 moments du mois
        # Typiquement début (1-5) et milieu (15-20) du mois
        early_month = sum(1 <= d <= 5 for d in days)
        mid_month = sum(13 <= d <= 18 for d in days)
        
        total_dates = len(days)
        
        # Si environ la moitié des dates sont en début et l'autre moitié en milieu
        if (early_month > total_dates * 0.4 and mid_month > total_dates * 0.4):
            return to_literal('SM') if literal else 'SM'
        
        return None


# Instance globale pour faciliter l'utilisation
_detector = FrequencyDetector()

# Fonctions de commodité pour la détection de fréquence
# Fonction de détection de la fréquence d'une série
def detect_frequency(data: Union[pd.Series, pd.DataFrame],
                    time_col: Optional[str] = None,
                    panel_cols: Optional[List[str]] = None,
                    literal: bool = False,
                    check_consistency: bool = False,
                    consistency_mode: str = 'modal',
                    strict: bool = True) -> Optional[Union[FrequencyType, UserFrequencyType, Dict]]:
    """Detect the frequency of time series data using FrequencyDetector.

    This function handles both Series and DataFrame inputs. For Series with MultiIndex,
    it returns a dictionary mapping panel IDs to frequencies. For DataFrames, it can
    detect frequencies for all columns and optionally return a consistent frequency.

    Args:
        data: Time series data (Series or DataFrame) with datetime index or MultiIndex
        time_col: Name of the time column (if None, uses index). Only for DataFrame.
        panel_cols: List of columns identifying panel dimensions. Only for DataFrame.
        literal: Boolean indicating whether to return explicit literal frequency expression
        check_consistency: If True, check frequency consistency across panel groups or columns
        consistency_mode: Mode for determining consistent frequency ('modal' or 'highest')
            - 'modal': Returns the most common frequency (default)
            - 'highest': Returns the highest frequency (most granular)
        strict: If True, all frequencies must be identical (used with check_consistency)

    Returns:
        For Series without MultiIndex: Detected frequency as string, or None if detection fails
        For Series with MultiIndex: Dictionary mapping panel_id to frequencies, or single frequency if check_consistency=True
        For DataFrame without check_consistency: Dictionary mapping columns to frequencies
        For DataFrame with check_consistency: Single consistent frequency or None

    Examples:
        >>> import pandas as pd
        >>> # Simple series detection
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> detect_frequency(series)
        'D'

        >>> # Series with MultiIndex (panel data)
        >>> idx = pd.MultiIndex.from_arrays([
        ...     ['A', 'A', 'A', 'B', 'B', 'B'],
        ...     pd.date_range('2023-01-01', periods=3, freq='D').tolist() * 2
        ... ], names=['panel_id', 'date'])
        >>> series = pd.Series([1, 2, 3, 4, 5, 6], index=idx)
        >>> detect_frequency(series)
        {'A': 'D', 'B': 'D'}

        >>> # Series with MultiIndex and consistency check
        >>> detect_frequency(series, check_consistency=True)
        'D'

        >>> # DataFrame detection with consistency check (modal)
        >>> df = pd.DataFrame({'a': series, 'b': series})
        >>> detect_frequency(df, check_consistency=True, consistency_mode='modal')
        'D'
    """
    # Cas 1: Entrée Series
    if isinstance(data, pd.Series):
        if time_col is not None or panel_cols:
            raise ValueError("time_col and panel_cols parameters are only valid for DataFrame inputs")

        # Détection de la fréquence (peut retourner un dict pour MultiIndex)
        freq_result = _detector.detect_frequency(data, literal)

        # Si check_consistency demandé et résultat est un dict (MultiIndex détecté)
        if check_consistency and isinstance(freq_result, dict):
            # Application de la logique de validation de consistance
            if consistency_mode == 'modal':
                # Mode modal: fréquence la plus commune
                _, frequency = _detector.validate_frequency_consistency(frequency_map=freq_result, strict=strict)
                return frequency
            elif consistency_mode == 'highest':
                # Mode highest: fréquence la plus haute (plus granulaire)
                return _get_highest_frequency(freq_result)
            else:
                raise ValueError(f"Invalid consistency_mode: {consistency_mode}. Must be 'modal' or 'highest'")

        # Sinon, retour direct du résultat
        return freq_result

    # Cas 2: Entrée DataFrame
    elif isinstance(data, pd.DataFrame):
        return detect_dataset_frequency(
            df=data,
            time_col=time_col,
            panel_cols=panel_cols,
            literal=literal,
            check_consistency=check_consistency,
            consistency_mode=consistency_mode,
            strict=strict
        )

    else:
        raise ValueError("Input must be a pandas Series or DataFrame")

# Fonction de détection de la fréquence d'un jeu de données
def detect_dataset_frequency(df: pd.DataFrame,
                           time_col: Optional[str] = None,
                           panel_cols: Optional[List[str]] = None,
                           literal: bool = False,
                           check_consistency: bool = False,
                           consistency_mode: str = 'modal',
                           strict: bool = True) -> Union[Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]], FrequencyType, UserFrequencyType]:
    """Detect frequencies for all series in a dataset using FrequencyDetector.

    Args:
        df: DataFrame containing time series data
        time_col: Name of the time column (if None, uses index)
        panel_cols: List of columns identifying panel dimensions
        literal: Boolean indicating whether to return explicit literal frequency expression
        check_consistency: If True, check the frequency consistency across columns
        consistency_mode: Mode for determining consistent frequency ('modal' or 'highest')
            - 'modal': Returns the most common frequency (default)
            - 'highest': Returns the highest frequency (most granular)
        strict: If True, all frequencies must be identical

    Returns:
        Dictionary mapping column names (or (panel_id, column) tuples) to frequencies or consistent frequency
    """
    # Détection des fréquences
    frequency_map = _detector.detect_dataset_frequency(df, time_col, panel_cols, literal)
    # Vérification de la consistence des fréquences si demandé
    if check_consistency :
        if consistency_mode == 'modal':
            # Mode modal: fréquence la plus commune
            _, frequency = _detector.validate_frequency_consistency(frequency_map=frequency_map, strict=strict)
        elif consistency_mode == 'highest':
            # Mode highest: fréquence la plus haute (plus granulaire)
            frequency = _get_highest_frequency(frequency_map)
        else:
            raise ValueError(f"Invalid consistency_mode: {consistency_mode}. Must be 'modal' or 'highest'")
        return frequency
    else :
        return frequency_map

# Fonction de détection de la fréquence d'un index
def detect_index_frequency(
    index: pd.DatetimeIndex
) -> FrequencyType:
    """Detect and parse DatetimeIndex frequency into components.

    Analyzes a DatetimeIndex to detect its frequency and parses it into
    three components: the base frequency code, the position indicator
    (S for start, E for end), and an optional suffix (e.g., DEC for
    quarterly data anchored to December).

    Args:
        index: DatetimeIndex to analyze. Must have at least 2 observations
            and a regular frequency pattern.

    Returns:
        Tuple of (frequency_code, position, suffix) where:
        - frequency_code: Pandas frequency indicator ('D', 'M', 'Q', etc.)
        - position: Position indicator ('S' for start, 'E' for end, or None)
        - suffix: Anchor suffix (e.g., 'DEC', 'JAN', or None)

    Raises:
        ValueError: If frequency cannot be detected or if the index has
            insufficient observations (<2) or irregular spacing.

    Examples:
        >>> import pandas as pd
        >>> # Daily frequency
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('D', None, None)
        >>>
        >>> # Monthly start
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('M', 'S', None)
        >>>
        >>> # Quarterly end with December anchor
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='QE-DEC')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('Q', 'E', 'DEC')
    """
    # Inférence de l'index
    freq = index.inferred_freq

    # Parsing de la fréquence
    return normalize_frequency(frequency=freq, return_format='base')

# Fonction de détection de la fréquence associée à un index
def detect_and_parse_index_frequency(
    index: pd.DatetimeIndex
) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect and parse DatetimeIndex frequency into components.

    Analyzes a DatetimeIndex to detect its frequency and parses it into
    three components: the base frequency code, the position indicator
    (S for start, E for end), and an optional suffix (e.g., DEC for
    quarterly data anchored to December).

    Args:
        index: DatetimeIndex to analyze. Must have at least 2 observations
            and a regular frequency pattern.

    Returns:
        Tuple of (frequency_code, position, suffix) where:
        - frequency_code: Pandas frequency indicator ('D', 'M', 'Q', etc.)
        - position: Position indicator ('S' for start, 'E' for end, or None)
        - suffix: Anchor suffix (e.g., 'DEC', 'JAN', or None)

    Raises:
        ValueError: If frequency cannot be detected or if the index has
            insufficient observations (<2) or irregular spacing.

    Examples:
        >>> import pandas as pd
        >>> # Daily frequency
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('D', None, None)
        >>>
        >>> # Monthly start
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('M', 'S', None)
        >>>
        >>> # Quarterly end with December anchor
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='QE-DEC')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('Q', 'E', 'DEC')
    """
    # Inférence de la fréquence de l'index
    freq = index.inferred_freq

    # Parsing de la fréquence
    return parse_frequency(frequency_str=freq)

# Fonction auxiliaire d'extraction de la fréquence la plus haute d'un dictionnaire de fréquences
def _get_highest_frequency(frequency_map: Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]) -> Optional[Union[FrequencyType, UserFrequencyType]]:
    """Get the highest (most granular) frequency from a frequency map.

    Args:
        frequency_map: Dictionary mapping columns to frequencies

    Returns:
        Highest frequency or None if map is empty

    Examples:
        >>> freq_map = {'col1': 'D', 'col2': 'M', 'col3': 'D'}
        >>> _get_highest_frequency(freq_map)
        'D'
    """
    # Vérification que le dictionnaire n'est pas vide
    if not frequency_map:
        return None

    # Extraction des fréquences uniques
    unique_frequencies = set(frequency_map.values())

    # Si une seule fréquence, la retourner
    if len(unique_frequencies) == 1:
        return list(unique_frequencies)[0]

    # Normalisation des fréquences avec ancrages (ex: 'W-SUN' -> 'W', 'Q-DEC' -> 'Q')
    # Calcul de l'ordre de chaque fréquence et sélection de la plus haute (ordre le plus bas)
    freq_orders = {}
    for freq in unique_frequencies:
        try:
            freq_orders[freq] = get_frequency_order(freq)
        except ValueError:
            # Si la normalisation échoue, essayer avec la fréquence complète
            try:
                freq_orders[freq] = get_frequency_order(freq)
            except ValueError:
                # Si toujours échec, ignorer cette fréquence
                continue

    if not freq_orders:
        return None

    # Retour de la fréquence avec l'ordre le plus bas (plus granulaire)
    highest_freq = min(freq_orders.keys(), key=lambda x: freq_orders[x])

    return highest_freq