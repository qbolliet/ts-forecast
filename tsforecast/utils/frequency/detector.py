"""Frequency detection utilities for time series data.

This module provides the FrequencyDetector class to detect and validate frequencies
in time series and panel data, with primary reliance on pandas.infer_freq.
"""
# Importation des modules
# Module de base
import pandas as pd
import numpy as np
from typing import Dict, Literal, Optional, Union, Tuple, List

# Import des utilitaires de fréquence
from .utils import normalize_frequency
from .types import FrequencyType, UserFrequencyType
from ..parse.utils import build_frequency_string
from ...panel.utils import normalize_entity_key, detect_panel_structure, extract_time_series_from_multiindex

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
    def detect_time_series_frequency(
        self,
        series: pd.Series,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect the frequency of a single time series with DatetimeIndex.

        This method primarily uses pandas.infer_freq with extensions for frequencies
        not natively supported (like quarterly). It automatically drops NaN values
        before detection.

        Args:
            series: Time series data with datetime index
            return_format: Output format for the detected frequency:
                - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
                - 'with_position': Frequency with position (e.g. 'MS', 'QE')
                - 'full': Full pandas frequency string (e.g. 'QE-DEC')
                - 'components': Tuple of (base, position, suffix)

        Returns:
            Detected frequency in the requested format, or None if detection fails

        Raises:
            ValueError: If series has insufficient non-null observations or invalid index

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, np.nan, 4, 5], index=dates)
            >>> detector = FrequencyDetector()
            >>> detector.detect_time_series_frequency(series)
            'D'
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
            # Normalisation au format demandé
            result = normalize_frequency(frequency=inferred_freq, return_format=return_format)
            if result:
                return result
        except Exception:
            # En cas d'erreur avec infer_freq, continuer avec la détection manuelle
            pass

        # Extension pour les fréquences non supportées par infer_freq (retourne des codes bruts)
        extended_freq = self._extend_infer_freq(time_index)
        if extended_freq:
            # Normalisation au format demandé
            return normalize_frequency(frequency=extended_freq, return_format=return_format)

        return None

    # Méthode auxiliaire d'extraction de la fréquence d'une colonne
    def _detect_column_frequency(
        self,
        series: pd.Series,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Optional[Union[FrequencyType, UserFrequencyType]]:
        """Detect the frequency of a column (with automatic MultiIndex handling).

        Args:
            series: Series for which to detect the frequency
            return_format: Output format for the detected frequency

        Returns:
            Detected frequency or None if failed/insufficient observations
        """
        try:
            # Si la série a un MultiIndex, extraire la série temporelle simple
            if isinstance(series.index, pd.MultiIndex):
                series = extract_time_series_from_multiindex(series)
            return self.detect_time_series_frequency(series, return_format)
        except ValueError:
            # Pas assez d'observations, retourner None
            return None

    # Méthode de détection de la fréquence d'une série (simple ou avec MultiIndex)
    def detect_frequency(
        self,
        series: pd.Series,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Optional[Union[FrequencyType, UserFrequencyType, Dict[tuple, Union[FrequencyType, UserFrequencyType]]]]:
        """Detect the frequency of a series (simple or with MultiIndex for panel data).

        This method handles both simple time series and panel data with MultiIndex.
        For MultiIndex, it groups by all levels except the last (assumed to be the date)
        and detects frequency for each panel group.

        Args:
            series: Time series data with DatetimeIndex or MultiIndex
            return_format: Output format for the detected frequency:
                - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
                - 'with_position': Frequency with position (e.g. 'MS', 'QE')
                - 'full': Full pandas frequency string (e.g. 'QE-DEC')
                - 'components': Tuple of (base, position, suffix)

        Returns:
            - For simple series: Detected frequency as string, or None if detection fails
            - For MultiIndex series: Dictionary mapping panel_id to frequencies.
              Panel ids are ALWAYS tuples, even for a single entity level. Entities
              for which frequency detection fails (e.g. fewer than
              ``min_observations`` dates) are still present in the dictionary,
              mapped to None, rather than being silently dropped. None is only
              returned if the series has no panel group at all.

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
            {('A',): 'D', ('B',): 'D'}
        """
        # Cas d'une série temporelle simple
        if not isinstance(series.index, pd.MultiIndex):
            return self.detect_time_series_frequency(series, return_format)

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

        # Groupby sur les niveaux de panel et détection pour chaque groupe.
        # Un niveau UNIQUE est passé sous forme scalaire (et non liste de
        # longueur 1)
        groupby_levels = panel_levels[0] if len(panel_levels) == 1 else panel_levels
        for panel_values, group_series in series.groupby(level=groupby_levels):
            # Création de l'identifiant du panel normalisé
            panel_id = normalize_entity_key(panel_values)

            # Extraction de la série temporelle simple et détection de la fréquence.
            # L'entité est TOUJOURS ajoutée au dictionnaire, avec None en cas
            # d'échec de détection
            temp_series = extract_time_series_from_multiindex(group_series)
            frequency_map[panel_id] = self._detect_column_frequency(temp_series, return_format)

        # Retour du dictionnaire (None si aucune entité, c'est-à-dire aucun groupe)
        return frequency_map if frequency_map else None

    # Méthode de détection des fréquences d'un jeu de données de panel
    def _detect_panel_frequencies(
        self,
        df: pd.DataFrame,
        panel_cols: List[str],
        panel_in_index: bool,
        time_col: Optional[str],
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a panel DataFrame.

        Args:
            df: DataFrame with panel structure
            panel_cols: List of panel columns
            panel_in_index: True if panel is in the index
            time_col: Name of the time column (or None if in the index)
            return_format: Output format for the detected frequency

        Returns:
            Dictionary mapping FLATTENED ``(entity..., column)`` tuples to
            frequencies: the entity part is spliced into the key rather than
            nested, so a single-level panel yields ``('FR', 'gdp')`` and a
            two-level panel ``('FR', 'manufacturing', 'gdp')``. Use
            :func:`tsforecast.panel.utils.split_variable_key` to split a key
            back into its ``(entity_tuple, column)`` parts. Keys for which
            frequency detection fails (e.g. fewer than ``min_observations``
            dates) are still present, mapped to None.
        """
        # Initialisation du dictionnaire des fréquences
        frequency_map = {}

        if panel_in_index:
            # Groupby par niveaux d'index. Un niveau UNIQUE est passé sous
            # forme scalaire (et non liste de longueur 1)
            groupby_levels = panel_cols[0] if len(panel_cols) == 1 else panel_cols
            for panel_values, group_df in df.groupby(level=groupby_levels):
                # Création de l'identifiant du panel
                panel_id = normalize_entity_key(panel_values)

                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    if col != time_col:
                        # Extraction de la série temporelle simple depuis le MultiIndex
                        series_with_multiindex = group_df[col]
                        simple_series = extract_time_series_from_multiindex(series_with_multiindex)

                        # Détection de la fréquence (None en cas d'échec)
                        frequency_map[panel_id + (col,)] = self._detect_column_frequency(
                            simple_series, return_format
                        )
        else:
            # Groupby par colonnes
            for panel_values, group_df in df.groupby(panel_cols):
                # Création de l'identifiant du panel
                panel_id = normalize_entity_key(panel_values)

                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    if col not in panel_cols and col != time_col:
                        # Détection de la fréquence (None en cas d'échec)
                        frequency_map[panel_id + (col,)] = self._detect_column_frequency(
                            group_df[col], return_format
                        )

        return frequency_map

    # Méthode auxiliaire de détection des fréquences pour les jeux de données qui sont des séries temporelles
    def _detect_time_series_frequencies(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a simple DataFrame (non-panel).

        Args:
            df: DataFrame
            time_col: Name of the time column (or None if in the index)
            return_format: Output format for the detected frequency

        Returns:
            Dictionary mapping column (or panel_id, column) to frequencies
        """
        # Initialisation du dictionnaire des fréquences
        frequency_map = {}

        # Traitement des séries temporelles
        for col in df.columns:
            if col != time_col:
                # Détection de la fréquence (peut retourner un dict si la colonne a un MultiIndex)
                freq_result = self.detect_frequency(df[col], return_format)

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
    def detect_dataset_frequency(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for all series in a dataset.

        This method handles both simple DataFrames and panel data. Panel structure can be
        specified either via panel_cols or detected automatically from a MultiIndex.

        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column (if None, uses index)
            panel_cols: List of columns identifying panel dimensions. If None and index is
                MultiIndex with at least 2 levels, automatically extracts panel structure
            return_format: Output format for the detected frequency:
                - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
                - 'with_position': Frequency with position (e.g. 'MS', 'QE')
                - 'full': Full pandas frequency string (e.g. 'QE-DEC')
                - 'components': Tuple of (base, position, suffix)

        Returns:
            Dictionary mapping column names (or (panel_id, column) tuples) to frequencies

        Raises:
            ValueError: If time_col is specified but not found in df.columns

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
        # Préparation de l'index temporel si spécifié. Une time_col absente des
        # colonnes lève une erreur
        if time_col is not None:
            if time_col not in df.columns:
                raise ValueError(
                    f"time_col '{time_col}' not found in DataFrame columns: {list(df.columns)}"
                )
            df = df.set_index(time_col)

        # Détection de la structure panel (dans l'index ou les colonnes)
        panel_cols, panel_in_index = detect_panel_structure(df, panel_cols)

        # Détermination si les données sont en panel
        is_panel = panel_cols is not None and len(panel_cols) > 0

        # Détection des fréquences selon le type de structure
        if is_panel:
            return self._detect_panel_frequencies(df, panel_cols, panel_in_index, time_col, return_format)
        else:
            return self._detect_time_series_frequencies(df, time_col, return_format)

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
    def _extend_infer_freq(self, time_index: pd.DatetimeIndex) -> Optional[FrequencyType]:
        """Extend frequency detection for frequencies not supported by pandas.infer_freq.

        This method provides custom detection for frequencies like quarterly, business daily,
        semi-monthly, and sub-daily frequencies that pandas.infer_freq doesn't always detect
        reliably.

        Args:
            time_index: Sorted datetime index

        Returns:
            Raw frequency code in pandas format or None

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
            return self._detect_intraday_frequency(modal_seconds=modal_seconds)
        else:
            # Conversion en jours pour les fréquences >= journalières
            modal_days = modal_diff.days
            return self._detect_day_frequency(time_index=time_index, modal_days=modal_days)

    # Méthode auxiliaire de détection des basses fréquences en jours
    def _detect_day_frequency(self, time_index: pd.DatetimeIndex, modal_days: float) -> Optional[FrequencyType]:
        """Detect day-level frequencies (daily, weekly, monthly, quarterly, annual).

        Args:
            time_index: Sorted datetime index
            modal_days: Modal time difference in days between consecutive observations

        Returns:
            Raw frequency code or None if no match found. For period-based
            frequencies (monthly, quarterly, annual), the position (start/end)
            is appended when it can be detected on ``time_index`` (e.g. 'MS'
            rather than bare 'M'), so callers requesting 'with_position',
            'full' or 'components' formats get a positioned frequency instead
            of having to re-detect it themselves.
        """
        # Détection basée sur le nombre de jours modal
        if modal_days == 1:
            # Distinction entre jours calendaires et jours ouvrés
            return self._detect_daily_frequency(time_index=time_index)
        elif modal_days == 7:
            return 'W'
        elif 13 <= modal_days <= 16:
            # Fréquence semi-mensuelle (environ 2 fois par mois)
            return self._detect_semi_monthly_frequency(time_index=time_index)
        elif 28 <= modal_days <= 31:
            return self._with_detected_position('M', time_index)
        elif 89 <= modal_days <= 92:
            return self._with_detected_position('Q', time_index)
        elif 365 <= modal_days <= 366:
            return self._with_detected_position('Y', time_index)

        return None

    # Méthode auxiliaire d'ajout de la position détectée à une fréquence de base
    @staticmethod
    def _with_detected_position(base_freq: FrequencyType, time_index: pd.DatetimeIndex) -> FrequencyType:
        """Combine a base frequency with its detected start/end position.

        ``pd.infer_freq`` already reports positioned codes (``'MS'``, ``'QE'``,
        ...) when it succeeds ; this only concerns the manual fallback path
        (:meth:`_detect_day_frequency`), which otherwise has no notion of
        position at all.

        Args:
            base_freq: Base frequency code ('M', 'Q' or 'Y').
            time_index: Sorted datetime index the frequency was detected on.

        Returns:
            ``base_freq`` combined with its detected position (e.g. ``'MS'``),
            or ``base_freq`` unchanged if the position is ambiguous.
        """
        position = FrequencyDetector._detect_period_position(base_freq, time_index)
        if position is None:
            return base_freq
        return build_frequency_string(base_freq, position)

    # Méthode auxiliaire de détection de la position (début/fin de période) d'un index
    @staticmethod
    def _detect_period_position(base_freq: FrequencyType, time_index: pd.DatetimeIndex) -> Optional[str]:
        """Detect whether dates are anchored at period start or period end.

        Args:
            base_freq: Base frequency code ('M', 'Q' or 'Y').
            time_index: Sorted datetime index to inspect.

        Returns:
            ``'S'`` if every date falls on its period's start, ``'E'`` if
            every date falls on its period's end, ``None`` if mixed/neither
            (or if ``base_freq`` isn't one of the period-based frequencies).
        """
        # Propriétés vectorisées de pandas pour chaque fréquence période
        position_checks = {
            'M': ('is_month_start', 'is_month_end'),
            'Q': ('is_quarter_start', 'is_quarter_end'),
            'Y': ('is_year_start', 'is_year_end'),
        }
        if base_freq not in position_checks:
            return None

        start_attr, end_attr = position_checks[base_freq]
        if getattr(time_index, start_attr).all():
            return 'S'
        if getattr(time_index, end_attr).all():
            return 'E'
        return None

    # Méthode de détection des fréquences infrajournalières
    def _detect_intraday_frequency(self, modal_seconds: float) -> Optional[FrequencyType]:
        """Detect intraday frequency (hourly, minute, second).

        Args:
            modal_seconds: Modal time difference in seconds

        Returns:
            Raw pandas frequency code for intraday frequency or None

        Notes:
            Detects hourly ('h'), minute ('min'), and second ('s') frequencies.
        """
        # Tolérance de 5% pour gérer les petites variations
        tolerance = 0.05

        # Fréquence horaire
        if abs(modal_seconds - 3600) / 3600 < tolerance:
            return 'h'

        # Fréquence par minute
        if abs(modal_seconds - 60) / 60 < tolerance:
            return 'min'

        # Fréquence par seconde
        if abs(modal_seconds - 1) / 1 < tolerance:
            return 's'

        # Sous-seconde (millisecondes, microsecondes, nanosecondes)
        if modal_seconds < 1:
            if modal_seconds >= 0.001:
                return 'ms'
            elif modal_seconds >= 0.000001:
                return 'us'
            else:
                return 'ns'

        return None

    # Méthode auxiliaire de détection des fréquences journalières
    def _detect_daily_frequency(self, time_index: pd.DatetimeIndex) -> FrequencyType:
        """Detect whether daily frequency is calendar daily or business daily.

        Args:
            time_index: Sorted datetime index

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
                return 'B'

        return 'D'

    # Méthode auxiliaire de détection des fréquences semi-mensuelles
    def _detect_semi_monthly_frequency(self, time_index: pd.DatetimeIndex) -> Optional[FrequencyType]:
        """Detect semi-monthly frequency (twice per month).

        Args:
            time_index: Sorted datetime index

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
            return 'SM'

        return None
