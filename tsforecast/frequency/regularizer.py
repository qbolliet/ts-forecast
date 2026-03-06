"""Index regularization utilities for time series data.

This module provides the IndexRegularizer class and utility functions to detect
and fix irregular time series indices (e.g., gaps, mixed frequencies) by reindexing
onto a regular date_range.
"""
# Importation des modules
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# Import des utilitaires internes
from .detector import FrequencyDetector, detect_frequency, _get_highest_frequency, detect_and_parse_index_frequency
from ..utils.frequency import to_pandas_freq, normalize_frequency, FrequencyType, UserFrequencyType


# Classe de régularisation d'index temporel
class IndexRegularizer:
    """Detect and fix irregular time series indices.

    Provides methods to check whether a DatetimeIndex (or MultiIndex with dates)
    is regular, and to regularize it by filling temporal gaps with NaN rows.

    Attributes:
        min_observations (int): Minimum observations required for frequency detection.

    Examples:
        >>> import pandas as pd
        >>> regularizer = IndexRegularizer()
        >>> dates = pd.to_datetime(['2023-01-01', '2023-03-01', '2023-04-01'])
        >>> series = pd.Series([1, 3, 4], index=dates)
        >>> regularizer.is_regular(series)
        False
        >>> result = regularizer.regularize(series)
        >>> len(result)
        4
    """

    def __init__(self, min_observations: int = 2):
        """Initialize the IndexRegularizer.

        Args:
            min_observations: Minimum number of observations required for
                frequency detection.
        """
        self.min_observations = min_observations
        self._detector = FrequencyDetector(min_observations=min_observations)

    # ------------------------------------------------------------------ #
    #  Détection de la régularité                                         #
    # ------------------------------------------------------------------ #

    def is_regular(
        self,
        data: Union[pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        per_entity: bool = False,
    ) -> Union[bool, Dict[tuple, bool]]:
        """Check whether the temporal index is regular (no gaps).

        A regular index is one where ``pd.infer_freq`` returns a non-None value.

        Args:
            data: Time series or panel data to check.
            time_col: If provided, the column containing timestamps (will be
                set as index for the check).
            panel_cols: Columns identifying panel entities. When provided, the
                data is treated as panel data.
            per_entity: If True and data is panel, return a dict mapping each
                entity key to its regularity boolean. If False, return a single
                bool (True only if every entity is regular **and** all share the
                same frequency).

        Returns:
            bool for time series or panel with ``per_entity=False``.
            Dict[tuple, bool] for panel with ``per_entity=True``.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='MS')
            >>> series = pd.Series(range(5), index=dates)
            >>> IndexRegularizer().is_regular(series)
            True
        """
        # Préparation : copie de travail + mise en index si nécessaire
        data = self._prepare_data(data, time_col, panel_cols)

        # Série temporelle simple (DatetimeIndex)
        if not isinstance(data.index, pd.MultiIndex):
            return self._is_regular_single(data.index)

        # Panel (MultiIndex)
        return self._is_regular_panel(data, per_entity)

    # ------------------------------------------------------------------ #
    #  Régularisation                                                     #
    # ------------------------------------------------------------------ #

    def regularize(
        self,
        data: Union[pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        per_entity: bool = False,
        target_frequency: Optional[str] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize the temporal index by filling gaps with NaN rows.

        For panel data each entity keeps its own date range (min/max); only
        internal gaps are filled.

        Args:
            data: Time series or panel data to regularize.
            time_col: Column containing timestamps. If provided, it is set as
                index before processing and restored as a column afterwards.
            panel_cols: Columns identifying panel entities.
            per_entity: For panel data — if True, the target frequency is
                detected independently per entity; if False, a single global
                frequency is used for every entity.
            target_frequency: Explicit target frequency (e.g. ``'MS'``,
                ``'monthly'``). If None, the frequency is auto-detected.

        Returns:
            Regularized data with the same type as the input.

        Raises:
            ValueError: If duplicated timestamps are found.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.to_datetime(['2023-01-01', '2023-03-01'])
            >>> series = pd.Series([1, 3], index=dates)
            >>> result = IndexRegularizer().regularize(series)
            >>> len(result)
            3
        """
        # Sauvegarde de la structure originale
        had_time_col = time_col is not None

        # Préparation : copie de travail + mise en index si nécessaire
        data = self._prepare_data(data, time_col, panel_cols)

        # Vérification des doublons
        if data.index.duplicated().any():
            raise ValueError("Duplicated timestamps found in the index.")

        # Tri de l'index
        data = data.sort_index()

        # Série temporelle simple
        if not isinstance(data.index, pd.MultiIndex):
            freq = self._resolve_frequency_ts(data, target_frequency)
            if freq is None:
                # Impossible de détecter la fréquence → retour inchangé
                result = data
            else:
                result = self._regularize_single(data, freq)
        else:
            # Panel
            result = self._regularize_panel(data, per_entity, target_frequency)

        # Restauration de time_col en colonne si nécessaire
        if had_time_col:
            if isinstance(result.index, pd.MultiIndex):
                # Le dernier niveau est la date
                result = result.reset_index(level=-1)
            else:
                result = result.reset_index()
                # Renommage si le nom d'index ne correspond pas à time_col
                if result.columns[0] != time_col:
                    result = result.rename(columns={result.columns[0]: time_col})

        return result

    # ------------------------------------------------------------------ #
    #  Helpers privés                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_data(
        self,
        data: Union[pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Prepare a working copy with proper index structure.

        Args:
            data: Original data.
            time_col: Column to set as (last level of) the index.
            panel_cols: Panel columns to include as index levels.

        Returns:
            Copy of data with DatetimeIndex or MultiIndex(entities..., date).
        """
        data = data.copy()

        # Construction de l'index à partir des colonnes spécifiées
        if time_col is not None or panel_cols is not None:
            idx_cols = []
            if panel_cols is not None:
                idx_cols.extend(panel_cols)
            if time_col is not None:
                idx_cols.append(time_col)
            if idx_cols:
                data = data.set_index(idx_cols)

        return data

    @staticmethod
    def _is_regular_single(index: pd.DatetimeIndex) -> bool:
        """Check regularity of a single DatetimeIndex.

        Args:
            index: DatetimeIndex to check.

        Returns:
            True if ``pd.infer_freq`` succeeds.
        """
        if len(index) < 2:
            return False
        try:
            return pd.infer_freq(index) is not None
        except (TypeError, ValueError):
            return False

    def _is_regular_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        per_entity: bool,
    ) -> Union[bool, Dict[tuple, bool]]:
        """Check regularity for panel data (MultiIndex).

        Args:
            data: Panel data with MultiIndex.
            per_entity: Return per-entity dict or global bool.

        Returns:
            Dict or bool.
        """
        date_level = data.index.nlevels - 1
        entity_index = data.index.droplevel(date_level)
        results: Dict[tuple, bool] = {}
        detected_freqs = set()

        for entity_key in entity_index.unique():
            mask = entity_index == entity_key
            dates = data.index.get_level_values(date_level)[mask]

            regular = self._is_regular_single(dates)
            key = entity_key if isinstance(entity_key, tuple) else (entity_key,)
            results[key] = regular

            if regular:
                detected_freqs.add(pd.infer_freq(dates))

        if per_entity:
            return results

        # Global : toutes régulières ET même fréquence
        all_regular = all(results.values())
        same_freq = len(detected_freqs) <= 1
        return all_regular and same_freq

    def _resolve_frequency_ts(
        self,
        data: Union[pd.Series, pd.DataFrame],
        target_frequency: Optional[str],
    ) -> Optional[str]:
        """Resolve the pandas frequency string for a time series.

        Args:
            data: Time series data.
            target_frequency: User-supplied frequency or None.

        Returns:
            Pandas-compatible frequency string, or None.
        """
        if target_frequency is not None:
            # Préservation de la position start/end (ex: 'MS' reste 'MS')
            return normalize_frequency(target_frequency, return_format='full')

        # Auto-détection via detect_and_parse_index_frequency pour préserver la position
        try:
            parsed = detect_and_parse_index_frequency(data.index)
            # parsed = (base, position, suffix)  ex: ('M', 'S', None) → 'MS'
            base, position, suffix = parsed
            freq_str = base
            if position:
                freq_str += position
            if suffix:
                freq_str += f"-{suffix}"
            return freq_str
        except (ValueError, TypeError):
            pass

        # Fallback : détection par colonnes puis highest
        try:
            base_freq = None
            if isinstance(data, pd.DataFrame):
                freq_map = detect_frequency(data)
                if isinstance(freq_map, dict):
                    highest = _get_highest_frequency(freq_map)
                    if highest:
                        base_freq = highest
            elif isinstance(data, pd.Series):
                freq = detect_frequency(data)
                if freq is not None and not isinstance(freq, dict):
                    base_freq = freq

            if base_freq is not None:
                # Ajout de la position start/end inférée depuis les dates
                return self._add_position_from_dates(base_freq, data.index)
        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def _add_position_from_dates(base_freq: str, index: pd.DatetimeIndex) -> str:
        """Infer start/end position from actual dates and build a pandas freq string.

        For frequencies that distinguish start vs end (M, Q, Y), checks whether
        dates fall on the first day of the period (→ start) or last day (→ end).

        Args:
            base_freq: Base frequency code (e.g. 'M', 'Q', 'Y').
            index: DatetimeIndex to inspect.

        Returns:
            Pandas frequency string with position (e.g. 'MS', 'QE', 'ME').
        """
        normalized = normalize_frequency(base_freq)

        # Fréquences qui nécessitent une distinction start/end
        position_freqs = {'M', 'Q', 'Y', 'A'}
        if normalized not in position_freqs:
            return to_pandas_freq(base_freq)

        # Vérification : les dates tombent-elles en début de période ?
        dates = index
        if isinstance(index, pd.MultiIndex):
            dates = index.get_level_values(-1)

        if len(dates) == 0:
            return to_pandas_freq(base_freq)

        all_first_day = all(d.day == 1 for d in dates)

        if all_first_day:
            # Start position
            return f"{normalized}S"
        else:
            # End position
            return f"{normalized}E"

    def _regularize_single(
        self,
        data: Union[pd.Series, pd.DataFrame],
        target_frequency: str,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize a single time series (no panel dimension).

        Builds a regular ``date_range`` from min to max of the existing index
        and reindexes the data.

        Args:
            data: Time series data with DatetimeIndex.
            target_frequency: Pandas frequency string (e.g. ``'MS'``, ``'QE-DEC'``).

        Returns:
            Reindexed data with NaN for filled gaps.
        """
        new_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=target_frequency,
        )
        new_index.name = data.index.name
        return data.reindex(new_index)

    def _regularize_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        per_entity: bool,
        target_frequency: Optional[str],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize panel data entity by entity.

        Args:
            data: Panel data with MultiIndex.
            per_entity: If True, detect frequency per entity; if False, use a
                single global frequency for all entities.
            target_frequency: User-supplied target or None.

        Returns:
            Regularized panel data.
        """
        date_level = data.index.nlevels - 1
        entity_index = data.index.droplevel(date_level)
        entity_keys = entity_index.unique()

        # Fréquence globale (si per_entity=False)
        global_freq = None
        if not per_entity:
            if target_frequency is not None:
                global_freq = normalize_frequency(target_frequency, return_format='full')
            else:
                # Détection de la fréquence la plus haute sur l'ensemble du panel
                try:
                    parsed = detect_and_parse_index_frequency(data.index)
                    if isinstance(parsed, dict):
                        # Conversion des tuples parsés en fréquences string
                        freq_map = {}
                        for k, (base, position, suffix) in parsed.items():
                            f = base
                            if position:
                                f += position
                            if suffix:
                                f += f"-{suffix}"
                            freq_map[k] = f
                        global_freq = _get_highest_frequency(freq_map)
                    else:
                        base, position, suffix = parsed
                        global_freq = base
                        if position:
                            global_freq += position
                        if suffix:
                            global_freq += f"-{suffix}"
                except (ValueError, TypeError):
                    pass

                # Fallback : détection par entité puis fréquence la plus haute
                if global_freq is None:
                    entity_freqs = {}
                    for ek in entity_keys:
                        m = entity_index == ek
                        dates = data.index.get_level_values(date_level)[m]
                        entity_data_tmp = data[m]
                        if isinstance(entity_data_tmp, pd.DataFrame):
                            entity_data_tmp = entity_data_tmp.copy()
                            entity_data_tmp.index = dates
                        else:
                            entity_data_tmp = pd.Series(entity_data_tmp.values, index=dates)
                        f = self._resolve_frequency_ts(entity_data_tmp, None)
                        if f is not None:
                            key = ek if isinstance(ek, tuple) else (ek,)
                            entity_freqs[key] = f
                    if entity_freqs:
                        global_freq = _get_highest_frequency(entity_freqs)

        # Régularisation par entité
        parts = []
        level_names = data.index.names

        for entity_key in entity_keys:
            mask = entity_index == entity_key
            subset = data[mask]

            # Extraction du DatetimeIndex de cette entité
            dates = subset.index.get_level_values(date_level)

            # Construction d'un DataFrame/Series indexé par date uniquement
            if isinstance(subset, pd.DataFrame):
                entity_data = subset.copy()
                entity_data.index = dates
            else:
                entity_data = pd.Series(subset.values, index=dates, name=subset.name)

            # Résolution de la fréquence pour cette entité
            if per_entity:
                freq = self._resolve_frequency_ts(entity_data, target_frequency)
            else:
                freq = global_freq

            if freq is None:
                # Pas de fréquence détectable → garder tel quel
                parts.append(subset)
                continue

            # Régularisation
            regularized = self._regularize_single(entity_data, freq)

            # Reconstruction du MultiIndex
            key = entity_key if isinstance(entity_key, tuple) else (entity_key,)
            n = len(regularized)

            # Niveaux d'entité
            entity_levels = [np.full(n, k) for k in key]
            arrays = entity_levels + [regularized.index]

            new_mi = pd.MultiIndex.from_arrays(arrays, names=level_names)

            if isinstance(regularized, pd.DataFrame):
                regularized.index = new_mi
            else:
                regularized = pd.Series(regularized.values, index=new_mi, name=regularized.name)

            parts.append(regularized)

        if not parts:
            return data

        return pd.concat(parts)


# Instance singleton
_regularizer = IndexRegularizer()


# ------------------------------------------------------------------ #
#  Fonctions utilitaires                                              #
# ------------------------------------------------------------------ #

def is_regular(
    data: Union[pd.Series, pd.DataFrame],
    time_col: Optional[str] = None,
    panel_cols: Optional[List[str]] = None,
    per_entity: bool = False,
) -> Union[bool, Dict[tuple, bool]]:
    """Check whether the temporal index is regular (no gaps).

    Convenience wrapper around :meth:`IndexRegularizer.is_regular`.

    Args:
        data: Time series or panel data.
        time_col: Column containing timestamps.
        panel_cols: Panel entity columns.
        per_entity: Per-entity result for panel data.

    Returns:
        bool or Dict[tuple, bool].

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='MS')
        >>> is_regular(pd.Series(range(5), index=dates))
        True
    """
    return _regularizer.is_regular(data, time_col, panel_cols, per_entity)


def regularize(
    data: Union[pd.Series, pd.DataFrame],
    time_col: Optional[str] = None,
    panel_cols: Optional[List[str]] = None,
    per_entity: bool = False,
    target_frequency: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Regularize the temporal index by filling gaps with NaN rows.

    Convenience wrapper around :meth:`IndexRegularizer.regularize`.

    Args:
        data: Time series or panel data.
        time_col: Column containing timestamps.
        panel_cols: Panel entity columns.
        per_entity: Detect frequency per entity (panel only).
        target_frequency: Explicit target frequency.

    Returns:
        Regularized data.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.to_datetime(['2023-01-01', '2023-03-01'])
        >>> result = regularize(pd.Series([1, 3], index=dates))
        >>> len(result)
        3
    """
    return _regularizer.regularize(data, time_col, panel_cols, per_entity, target_frequency)
