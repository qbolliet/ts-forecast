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
from .utils import normalize_frequency, to_literal, FrequencyType, UserFrequencyType

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

    # Méthode de détection d'une série temporelle
    def detect_frequency(self, series: pd.Series, literal: bool=False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect the frequency of a time series.

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
            >>> detector.detect_frequency(series)
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
        if extended_freq and literal:
            # Conversion vers le format littéral
            return to_literal(extended_freq)
        elif extended_freq:
            return extended_freq

        return None

    # Méthode de détection de la fréquence d'un jeu de données
    def detect_dataset_frequency(self,
                               df: pd.DataFrame,
                               time_col: Optional[str] = None,
                               panel_cols: Optional[List[str]] = None,
                               literal: bool = False) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for all series in a dataset.

        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column (if None, uses index)
            panel_cols: List of columns identifying panel dimensions
            literal: Boolean indicating whether to return explicit literal frequency expression

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
        # Initialisation du dictionnaire résultat
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
                        # Détection de la fréquence
                        try:
                            freq = self.detect_frequency(group_df[col], literal)
                            # Ajout de la fréquence au dictionnaire résultat
                            if freq:
                                frequency_map[(panel_id, col)] = freq
                        except ValueError:
                            # Pas assez d'observations pour cette série
                            continue
        else:
            # Traitement des séries temporelles simples
            for col in df.columns:
                if col != time_col:
                    # Détection de la fréquence
                    try:
                        freq = self.detect_frequency(df[col], literal)
                        # Ajout de la fréquence au dictionnaire résultat
                        if freq:
                            frequency_map[col] = freq
                    except ValueError:
                        # Pas assez d'observations pour cette série
                        continue

        return frequency_map

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

        # Dans le cas où les fréquences 
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
    def _detect_day_frequency(self, time_index: pd.DatetimeIndex, modal_days: float, literal: bool = False) -> Optional[Union[FrequencyType, UserFrequencyType]] :
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
            Détecte les fréquences horaires ('h'), par minute ('min'), et par seconde ('s').
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
            Vérifie si les dates tombent uniquement sur des jours ouvrés (lundi-vendredi).
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
            Vérifie si les dates correspondent à des occurrences bi-mensuelles
            (typiquement 1er et 15 du mois, ou début et milieu de mois).
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
def detect_frequency(series: pd.Series, literal: bool = False) -> Optional[Union[FrequencyType, UserFrequencyType]]:
    """Detect the frequency of a time series using FrequencyDetector.

    Args:
        series: Time series data with datetime index
        literal: Boolean indicating whether to return explicit literal frequency expression

    Returns:
        Detected frequency as user-friendly string, or None if detection fails
    """
    return _detector.detect_frequency(series, literal)

# Fonction de détection de la fréquence d'un jeu de données
def detect_dataset_frequency(df: pd.DataFrame,
                           time_col: Optional[str] = None,
                           panel_cols: Optional[List[str]] = None,
                           literal: bool = False,
                           check_consistency: bool = False,
                           strict: bool = True) -> Union[Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]], FrequencyType, UserFrequencyType]:
    """Detect frequencies for all series in a dataset using FrequencyDetector.

    Args:
        df: DataFrame containing time series data
        time_col: Name of the time column (if None, uses index)
        panel_cols: List of columns identifying panel dimensions
        literal: Boolean indicating whether to return explicit literal frequency expression
        check_consistency: If True, check the frequency consistency across columns 
        strict: If True, all frequencies must be identical

    Returns:
        Dictionary mapping column names (or (panel_id, column) tuples) to frequencies or consistent frequency
    """
    # Détection des fréquences
    frequency_map = _detector.detect_dataset_frequency(df, time_col, panel_cols, literal)
    # Vérification de la consistence des fréquences si demandé
    if check_consistency :
        _, frequency = _detector.validate_frequency_consistency(frequency_map=frequency_map, strict=strict)
        return frequency
    else :
        return frequency_map