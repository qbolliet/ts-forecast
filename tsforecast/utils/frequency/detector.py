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
from .utils import normalize_frequency, canonicalize_frequency
from .types import FrequencyType, UserFrequencyType
from .._constants import INTRADAY_UNITS, MONTH_ABBREVIATIONS, WEEKDAY_ABBREVIATIONS
from ..parse.utils import build_frequency_string
from ..validation.utils import _convert_to_datetime
from ...panel.utils import normalize_entity_key, detect_panel_structure, extract_time_series_from_multiindex

# Formats de sortie acceptés par toutes les méthodes de détection
_RETURN_FORMATS = ('base', 'with_position', 'full', 'components')

# Écart toléré entre un écart modal et un nombre entier de jours (changement d'heure
# d'un index localisé : 23 h ou 25 h entre deux minuits)
_DAY_TOLERANCE = pd.Timedelta(hours=1)


# Fonction auxiliaire de vérification du format de sortie demandé
def _check_return_format(return_format: str) -> None:
    """Reject an unknown ``return_format`` before any detection.

    Args:
        return_format: Requested output format.

    Raises:
        ValueError: If ``return_format`` is not one of 'base', 'with_position',
            'full', 'components'.
    """
    if return_format not in _RETURN_FORMATS:
        raise ValueError(
            f"Invalid return_format: {return_format}. "
            f"Must be one of: 'base', 'with_position', 'full', 'components'"
        )


# Fonction auxiliaire de vérification qu'un écart modal justifie un multiplicateur
def _is_dominant(is_modal: pd.Series) -> bool:
    """Tell whether the modal spacing is evidenced enough to carry a multiplier.

    The modal spacing must be observed at least twice (two dates alone always
    have a spacing, a repetition is needed to call it a frequency, as
    ``pd.infer_freq`` needs three dates) and be shared by strictly more than
    half of the spacings.

    Args:
        is_modal: Boolean series, one entry per spacing, True where the
            spacing equals the modal one.

    Returns:
        True if the modal spacing is repeated and in strict majority.
    """
    return bool(is_modal.sum() >= 2 and is_modal.mean() > 0.5)


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
        >>> dates = pd.date_range('2023-01-31', periods=12, freq='ME')
        >>> series = pd.Series(range(12), index=dates)
        >>> detector.detect_frequency(series)
        'M'
        >>> detector.detect_frequency(series, return_format='with_position')
        'ME'
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

        NaN values are dropped, then duplicated dates are merged and the dates
        sorted (a decreasing index is read as the increasing one).
        ``pandas.infer_freq`` is tried first; it only succeeds on a
        perfectly regular grid of at least three dates. Otherwise (two dates,
        gaps, irregular spacing) a heuristic fallback reads the **modal
        spacing** between consecutive dates:

        - calendar grids (every date at a month start, at a month end, or on
          the same day of the month) are read in months, which gives
          ``'MS'``, ``'QE-DEC'``, ``'YS-JUL'``, ``'2MS'`` (every two months),
          ``'2QS-JAN'`` (half-years), ...;
        - weekly grids keep their weekday anchor (``'W-MON'``, ``'2W-WED'``);
        - otherwise the spacing is expressed as a multiple of the largest
          unit that divides it (``'3D'``, ``'90min'``, ``'10ms'``);
        - semi-monthly grids give ``'SMS'`` / ``'SME'`` / ``'SM'``, consecutive
          business days ``'B'`` (only when the span holds a weekend: two
          consecutive weekdays alone are ``'D'``).

        A multiplied frequency (``'2W'``, ``'10D'``) is only reported when the
        modal spacing is observed **at least twice** and shared by **strictly
        more than half** of the spacings: two dates alone give a simple
        frequency (``'D'``, ``'MS'``, ``'W-MON'``) or None, never ``'45D'``,
        and an index spaced by 45 then 50 days has no frequency (``None``).

        On an irregular index, the result is thus the frequency of the
        dominant grid, not a proof of regularity: an index made of a monthly
        grid plus a few isolated earlier annual dates is ``'MS'``. This is
        what an imputation onto that grid needs; use
        :func:`tsforecast.frequency.is_regular` to test regularity itself.

        Args:
            series: Time series data with a datetime index. Strings parsable
                as dates are converted, ``Period`` labels are converted to the
                first instant of their period; numeric labels (years,
                ``RangeIndex``) are rejected.
            return_format: Output format for the detected frequency:
                - 'base': Base frequency code (e.g. 'M', 'Q', 'D'), without
                  multiplier
                - 'with_position': Frequency with position (e.g. 'MS', 'QE'),
                  without multiplier
                - 'full': Full pandas frequency string (e.g. 'QE-DEC', '2MS')
                - 'components': ParsedFrequency (base, position, suffix, multiplier)

        Returns:
            Detected frequency in the requested format, or None if no
            frequency can be read from the dates.

        Raises:
            ValueError: If series has insufficient non-null observations, if its
                index cannot be converted to datetime, or if ``return_format``
                is unknown

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, np.nan, 4, 5], index=dates)
            >>> detector = FrequencyDetector()
            >>> detector.detect_time_series_frequency(series)
            'D'

            >>> # Deux dates au lundi : repli heuristique, ancre conservée
            >>> monday = pd.date_range('2024-01-01', periods=2, freq='W-MON')
            >>> detector.detect_time_series_frequency(pd.Series([1, 2], index=monday), 'full')
            'W-MON'
        """
        # Vérification du format demandé, avant toute analyse des données
        _check_return_format(return_format)

        # Suppression des valeurs manquantes pour la détection
        clean_series = series.dropna()

        # Vérification que le nombre d'observations dans le série est supérieur au minimum requis
        if len(clean_series) < self.min_observations:
            raise ValueError(
                f"Series has only {len(clean_series)} non-null observations, "
                f"minimum required is {self.min_observations}"
            )

        # Conversion de l'index en dates : mêmes règles que la validation temporelle
        # (périodes à leur premier instant, étiquettes numériques refusées)
        time_index = _convert_to_datetime(clean_series.index)
        if time_index is None:
            raise ValueError(
                "Series index cannot be converted to datetime "
                "(numeric labels are not dates)"
            )

        # Dédoublonnage : un écart nul entre deux dates identiques n'est pas une
        # fréquence (sinon l'écart modal peut valoir 0 et être pris pour 'ns')
        time_index = time_index.unique()

        # Tri de l'index temporel pour assurer la cohérence
        if not time_index.is_monotonic_increasing:
            time_index = time_index.sort_values()

        # Utilisation principale de pandas.infer_freq (grille régulière d'au moins 3 dates)
        try:
            inferred_freq = pd.infer_freq(time_index)
        except (ValueError, TypeError):
            # Moins de 3 dates : détection par le repli heuristique
            inferred_freq = None
        if inferred_freq is not None:
            try:
                # Normalisation au format demandé, sous forme canonique ('QS-OCT' -> 'QS-JAN')
                return normalize_frequency(
                    frequency=canonicalize_frequency(inferred_freq), return_format=return_format
                )
            except ValueError:
                # Alias pandas non pris en charge ('BME', 'WOM-1MON', ...) : repli heuristique
                pass

        # Extension pour les fréquences non reconnues par infer_freq (chaîne pandas complète)
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

        Only a lack of observations maps to None: an index that is not a time
        index, or an unknown ``return_format``, still raises.

        Args:
            series: Series for which to detect the frequency
            return_format: Output format for the detected frequency

        Returns:
            Detected frequency, or None if the column has fewer than
            ``min_observations`` non-null values or no readable frequency

        Raises:
            ValueError: If the index cannot be converted to datetime or
                ``return_format`` is unknown
        """
        # Si la série a un MultiIndex, extraire la série temporelle simple
        if isinstance(series.index, pd.MultiIndex):
            series = extract_time_series_from_multiindex(series)

        # Pas assez d'observations : couple indétectable, et non erreur
        if series.count() < self.min_observations:
            return None

        return self.detect_time_series_frequency(series, return_format)

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
                - 'components': ParsedFrequency (base, position, suffix, multiplier)

        Returns:
            - For simple series: Detected frequency as string, or None if detection fails
            - For MultiIndex series: Dictionary mapping panel_id to frequencies.
              Panel ids are always tuples, even for a single entity level. Entities
              for which frequency detection fails (e.g. fewer than
              ``min_observations`` dates) are still present in the dictionary,
              mapped to None, rather than being silently dropped. None is only
              returned if the series has no panel group at all.

        Raises:
            ValueError: If a simple series has insufficient non-null
                observations, if an index cannot be converted to datetime, if
                a MultiIndex has fewer than 2 levels, or if ``return_format``
                is unknown

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
        # Vérification du format demandé, y compris pour un panel sans groupe
        _check_return_format(return_format)

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

        # Extraction des niveaux de panel (tous sauf le dernier), désignés par position
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
        panel_cols: List[Optional[str]],
        panel_in_index: bool,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a panel DataFrame.

        Args:
            df: DataFrame with panel structure (time column already moved to
                the index)
            panel_cols: List of panel columns, or of panel index level names
                (None for an unnamed level)
            panel_in_index: True if panel is in the index
            return_format: Output format for the detected frequency

        Returns:
            Dictionary mapping flattened ``(entity..., column)`` tuples to
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

        # Cas où les entités sont dans l'index
        if panel_in_index:
            # Niveaux d'entité désignés par position : un niveau sans nom (None)
            # ne peut pas être désigné par son nom dans un groupby. Les niveaux
            # auto-détectés sont les premiers de l'index, dans l'ordre
            level_names = list(df.index.names)
            levels = [
                level_names.index(col) if col is not None else position
                for position, col in enumerate(panel_cols)
            ]
            # Un niveau unique est passé sous forme scalaire (et non liste de
            # longueur 1)
            groupby_levels = levels[0] if len(levels) == 1 else levels
            for panel_values, group_df in df.groupby(level=groupby_levels):
                # Création de l'identifiant du panel
                panel_id = normalize_entity_key(panel_values)

                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    # Extraction de la série temporelle simple depuis le MultiIndex
                    simple_series = extract_time_series_from_multiindex(group_df[col])

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
                    if col not in panel_cols:
                        # Détection de la fréquence (None en cas d'échec)
                        frequency_map[panel_id + (col,)] = self._detect_column_frequency(
                            group_df[col], return_format
                        )

        return frequency_map

    # Méthode auxiliaire de détection des fréquences pour les jeux de données qui sont des séries temporelles
    def _detect_time_series_frequencies(
        self,
        df: pd.DataFrame,
        return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
    ) -> Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]:
        """Detect frequencies for a simple DataFrame (non-panel).

        Args:
            df: DataFrame (time column already moved to the index)
            return_format: Output format for the detected frequency

        Returns:
            Dictionary mapping each column (or flattened ``(entity..., column)``
            tuple for a column carrying a MultiIndex) to its frequency. Every
            column is present: one with fewer than ``min_observations`` values,
            or without readable frequency, is mapped to None, as a panel pair.
        """
        # Initialisation du dictionnaire des fréquences
        frequency_map = {}

        # Traitement des séries temporelles
        for col in df.columns:
            column = df[col]

            # Colonne à MultiIndex (panel_cols=[]) : une fréquence par entité
            if isinstance(column.index, pd.MultiIndex):
                freq_result = self.detect_frequency(column, return_format)
                # Fusion des résultats (identifiants d'entité toujours en tuple)
                for panel_id, freq in (freq_result or {}).items():
                    frequency_map[(*panel_id, col)] = freq
            else:
                # Colonne indétectable : présente, associée à None
                frequency_map[col] = self._detect_column_frequency(column, return_format)

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
        specified either via panel_cols or detected automatically from a MultiIndex
        (named or unnamed levels).

        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column (if None, uses index)
            panel_cols: List of columns identifying panel dimensions. If None and index is
                MultiIndex with at least 2 levels, automatically extracts panel structure
            return_format: Output format for the detected frequency:
                - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
                - 'with_position': Frequency with position (e.g. 'MS', 'QE')
                - 'full': Full pandas frequency string (e.g. 'QE-DEC')
                - 'components': ParsedFrequency (base, position, suffix, multiplier)

        Returns:
            Dictionary mapping column names (time series) or FLATTENED
            ``(entity..., column)`` tuples (panel, e.g. ``('FR', 'gdp')``) to
            frequencies. Every column / pair is present; one that has fewer
            than ``min_observations`` values, or no readable frequency, is
            mapped to None.

        Raises:
            ValueError: If time_col is specified but not found in df.columns,
                if the time index cannot be converted to datetime, or if
                ``return_format`` is unknown

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
        # Vérification du format demandé, y compris pour un jeu sans ligne
        _check_return_format(return_format)

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
            return self._detect_panel_frequencies(df, panel_cols, panel_in_index, return_format)
        else:
            return self._detect_time_series_frequencies(df, return_format)

    # Méthode de validation de la consistence de la fréquence un jeu de données
    def validate_frequency_consistency(self,
                                     frequency_map: Dict[str, str],
                                     strict: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate that all series have consistent frequencies.

        Undetectable entries (None) carry no frequency: they are ignored, so
        an entity observed once does not make the others inconsistent, and
        None is never returned as the common frequency.

        Args:
            frequency_map: Dictionary of detected frequencies
            strict: If True, all detected frequencies must be identical; if
                False, the most common detected frequency is returned

        Returns:
            Tuple of (is_consistent, common_frequency); ``(False, None)`` when
            the map holds no detected frequency at all

        Examples:
            >>> detector = FrequencyDetector()
            >>> freq_map = {'series1': 'D', 'series2': 'D', 'series3': None}
            >>> is_consistent, common_freq = detector.validate_frequency_consistency(freq_map)
            >>> is_consistent, common_freq
            (True, 'D')
        """
        # Fréquences effectivement détectées (les couples indétectables sont ignorés)
        detected = [freq for freq in frequency_map.values() if freq is not None]

        # Aucune fréquence détectée : pas de fréquence commune
        if not detected:
            return False, None

        # Détection des fréquences
        unique_frequencies = set(detected)

        # Cas des fréquences uniques
        if len(unique_frequencies) == 1:
            return True, detected[0]

        # Dans le cas où les fréquences ne sont pas unique et que la validation est stricte
        if strict:
            return False, None

        # En mode non-strict, recherche de la fréquence la plus commune
        freq_counts = {}
        # Parcours des valeurs
        for freq in detected:
            # Incrémentation de chaque fréquence
            freq_counts[freq] = freq_counts.get(freq, 0) + 1

        # Identification de la fréquence la plus commune
        most_common = max(freq_counts, key=freq_counts.get)

        return True, most_common

    # Méthode d'extension de la détection des fréquences au delà de ce que supporte infer_freq
    def _extend_infer_freq(self, time_index: pd.DatetimeIndex) -> Optional[str]:
        """Extend frequency detection for dates that pandas.infer_freq rejects.

        The frequency is read on the modal spacing between consecutive dates
        (see :meth:`detect_time_series_frequency` for the rules). A multiplied
        frequency requires the modal spacing to be shared by strictly more
        than half of the spacings.

        Args:
            time_index: Sorted datetime index without duplicates

        Returns:
            Full pandas frequency string (e.g. 'D', 'W-MON', '2MS', 'QE-DEC',
            '90min') or None
        """
        # Ne retourne rien si le nombre d'observations n'est pas suffisant
        if len(time_index) < self.min_observations:
            return None

        # Calcul des différences entre observations consécutives
        time_diffs = pd.Series(time_index).diff().dropna()

        # Une seule date : aucun écart, aucune fréquence
        if len(time_diffs) == 0:
            return None

        # Identification de la différence modale (la plus fréquente, la plus petite
        # en cas d'égalité) et de sa part parmi les écarts
        modal_diff = time_diffs.mode()[0]
        dominant = _is_dominant(time_diffs == modal_diff)

        # Nombre entier de jours, à une heure près (changement d'heure)
        n_days = round(modal_diff / pd.Timedelta(days=1))
        whole_days = n_days >= 1 and abs(modal_diff - pd.Timedelta(days=n_days)) <= _DAY_TOLERANCE

        # Détection des fréquences infra-journalières (ou non multiples du jour)
        if not whole_days:
            return self._detect_intraday_frequency(modal_diff=modal_diff, dominant=dominant)

        # Fréquences journalières ou plus basses
        return self._detect_day_frequency(time_index=time_index, n_days=n_days, dominant=dominant)

    # Méthode auxiliaire de détection des basses fréquences en jours
    def _detect_day_frequency(
        self,
        time_index: pd.DatetimeIndex,
        n_days: int,
        dominant: bool
    ) -> Optional[str]:
        """Detect day-level frequencies (daily, weekly, monthly, quarterly, annual).

        Args:
            time_index: Sorted datetime index without duplicates
            n_days: Modal spacing between consecutive dates, in whole days
            dominant: Whether the modal spacing is shared by strictly more
                than half of the spacings (required for a multiplied result)

        Returns:
            Full pandas frequency string, or None if no match found
        """
        # Grilles calendaires (mois, trimestre, année et leurs multiples)
        if n_days >= 28:
            calendar_freq = self._detect_calendar_frequency(time_index)
            if calendar_freq is not None:
                return calendar_freq

        # Distinction entre jours calendaires et jours ouvrés
        if n_days == 1:
            return self._detect_daily_frequency(time_index=time_index)

        # Fréquence semi-mensuelle (environ 2 fois par mois), avant les semaines :
        # le 1er et le 15 peuvent tomber le même jour de la semaine
        if 13 <= n_days <= 16:
            semi_monthly = self._detect_semi_monthly_frequency(time_index=time_index)
            if semi_monthly is not None:
                return semi_monthly

        # Semaines et leurs multiples, ancrées sur le jour de la semaine
        if n_days % 7 == 0:
            weekly = self._detect_weekly_frequency(time_index, n_days // 7, dominant)
            if weekly is not None:
                return weekly

        # Motif calendaire incohérent (jours du mois variables) : fréquence
        # période sans position ni ancre
        if 28 <= n_days <= 31:
            return 'M'
        if 89 <= n_days <= 92:
            return 'Q'
        if 365 <= n_days <= 366:
            return 'Y'

        # Multiple du jour, seulement si l'écart modal est majoritaire
        return build_frequency_string('D', multiplier=n_days) if dominant else None

    # Méthode auxiliaire de détection des grilles calendaires
    @staticmethod
    def _detect_calendar_frequency(time_index: pd.DatetimeIndex) -> Optional[str]:
        """Detect a monthly-based grid (months, quarters, years and multiples).

        A calendar grid has every date at a month start, every date at a
        month end, or every date on the same day of the month. Its step is
        the modal spacing counted in months: a multiple of 12 is annual, a
        multiple of 3 quarterly, anything else monthly (``pd.infer_freq``
        reports half-years as ``'2QS'``). Start / end grids are anchored on
        the month of their dates (canonical form for quarters).

        Args:
            time_index: Sorted datetime index without duplicates

        Returns:
            Full pandas frequency string (e.g. 'MS', '2QS-JAN', 'YE-JUN',
            'M' for a mid-month grid), or None if the dates are not a
            calendar grid or a multiplied step is not the majority
        """
        # Position commune des dates dans le mois
        if time_index.is_month_start.all():
            position = 'S'
        elif time_index.is_month_end.all():
            position = 'E'
        elif (time_index.day == time_index.day[0]).all():
            position = None
        else:
            return None

        # Écart modal en mois entre dates consécutives
        months = pd.Series(time_index.year * 12 + time_index.month)
        month_diffs = months.diff().dropna().astype(int)
        modal_months = int(month_diffs.mode()[0])
        dominant = _is_dominant(month_diffs == modal_months)

        # Fréquence de base et multiplicateur
        if modal_months % 12 == 0:
            base, multiplier = 'Y', modal_months // 12
        elif modal_months % 3 == 0:
            base, multiplier = 'Q', modal_months // 3
        else:
            base, multiplier = 'M', modal_months

        # Multiplicateur retenu seulement si l'écart modal est majoritaire
        if multiplier > 1 and not dominant:
            return None

        # Ancre (mois des dates) pour les trimestres et les années positionnés
        anchor = None
        if base in ('Q', 'Y') and position is not None:
            anchor = MONTH_ABBREVIATIONS[time_index[0].month - 1]

        return canonicalize_frequency(build_frequency_string(base, position, anchor, multiplier))

    # Méthode auxiliaire de détection des grilles hebdomadaires
    @staticmethod
    def _detect_weekly_frequency(
        time_index: pd.DatetimeIndex,
        n_weeks: int,
        dominant: bool
    ) -> Optional[str]:
        """Detect a weekly grid, anchored on its weekday.

        Args:
            time_index: Sorted datetime index without duplicates
            n_weeks: Modal spacing in weeks
            dominant: Whether the modal spacing is the majority (required
                for ``n_weeks > 1``)

        Returns:
            'W-MON', '2W-WED', ..., 'W' for a one-week spacing between
            different weekdays, or None for a multiplied spacing that is not
            the majority or whose dates do not share a weekday (a 91-day
            spacing between varying weekdays is a quarter, not 13 weeks)
        """
        # Jour de la semaine commun à toutes les dates
        weekdays = time_index.dayofweek
        same_weekday = bool((weekdays == weekdays[0]).all())

        # Multiple de la semaine : grille hebdomadaire véritable et écart majoritaire
        if n_weeks > 1 and not (same_weekday and dominant):
            return None

        # Ancre : jour de la semaine commun, s'il existe
        anchor = WEEKDAY_ABBREVIATIONS[weekdays[0]] if same_weekday else None

        return build_frequency_string('W', suffix=anchor, multiplier=n_weeks)

    # Méthode de détection des fréquences infrajournalières
    @staticmethod
    def _detect_intraday_frequency(modal_diff: pd.Timedelta, dominant: bool) -> Optional[str]:
        """Detect intraday frequency (hours, minutes, seconds and below, with multiples).

        Args:
            modal_diff: Modal time difference between consecutive dates
            dominant: Whether the modal spacing is the majority (required
                for a multiplied result)

        Returns:
            Pandas frequency string ('h', '30min', '36h', '10ms', ...) or None

        Notes:
            A spacing within 5% of one hour, minute or second is read as that
            unit (tolerance to small timestamp jitter); otherwise the spacing
            is expressed as a multiple of the largest unit dividing it.
        """
        # Tolérance de 5% pour gérer les petites variations autour d'une unité
        tolerance = 0.05
        modal_seconds = modal_diff.total_seconds()
        for unit, seconds in (('h', 3600), ('min', 60), ('s', 1)):
            if abs(modal_seconds - seconds) / seconds < tolerance:
                return unit

        # Multiple exact de la plus grande unité qui divise l'écart (la nanoseconde
        # divise tout écart : une unité est toujours trouvée)
        nanoseconds = modal_diff.value
        unit, unit_nanoseconds = next(
            (unit, size) for unit, size in INTRADAY_UNITS if nanoseconds % size == 0
        )
        multiplier = nanoseconds // unit_nanoseconds
        if multiplier > 1 and not dominant:
            return None
        return build_frequency_string(unit, multiplier=multiplier)

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
            'SMS' if dates fall exactly on the pandas semi-month start grid
            (1st and 15th), 'SME' if they fall exactly on the semi-month end
            grid (15th and month end), 'SM' for a looser semi-monthly pattern,
            None otherwise

        Notes:
            Checks if dates correspond to bi-monthly occurrences
            (typically 1st and 15th of the month, or beginning and middle of the month).
        """
        # Extraction des jours du mois
        days = time_index.day

        # Grilles natives pandas : position détectable sans ambiguïté
        if len(days) > 0 and np.isin(days, (1, 15)).all():
            return 'SMS'
        if len(days) > 0 and ((days == 15) | time_index.is_month_end).all():
            return 'SME'

        # Vérification si les dates sont regroupées autour de 2 moments du mois
        # Typiquement début (1-5) et milieu (15-20) du mois
        early_month = sum(1 <= d <= 5 for d in days)
        mid_month = sum(13 <= d <= 18 for d in days)

        total_dates = len(days)

        # Si environ la moitié des dates sont en début et l'autre moitié en milieu
        if (early_month > total_dates * 0.4 and mid_month > total_dates * 0.4):
            return 'SM'

        return None
