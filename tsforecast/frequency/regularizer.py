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
from .detector import FrequencyDetector, detect_frequency, _get_highest_frequency, detect_index_frequency
from ..utils.frequency import normalize_frequency
from ..panel.utils import normalize_entity_key


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

    # Initialisation
    def __init__(self, min_observations: int = 2):
        """Initialize the IndexRegularizer.

        Args:
            min_observations: Minimum number of observations required for
                frequency detection.
        """
        # Instanciation des attributs
        self.min_observations = min_observations
        # Initialisation d'un détecteur de fréquence
        self._detector = FrequencyDetector(min_observations=min_observations)

    #  Méthode de détection de la régularité
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
        # Préparation des données : copie et mise en index si nécessaire
        data = self._prepare_data(data, time_col, panel_cols)

        # Série temporelle simple (DatetimeIndex)
        if not isinstance(data.index, pd.MultiIndex):
            return self._is_regular_ts(data.index)

        # Panel (MultiIndex)
        return self._is_regular_panel(data.index, per_entity)

    #  Méthode de régularisation
    def regularize(
        self,
        data: Union[pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        per_entity: bool = False,
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
            # Détection de la fréquence de la série temporelle
            freq = self._resolve_frequency_ts(data)
            if freq is None:
                # Impossible de détecter la fréquence → retour inchangé
                result = data
            else:
                result = self._regularize_ts(data, freq)
        else:
            # Panel
            result = self._regularize_panel(data, per_entity)

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

    # Méthode auxiliaire de préparation des données
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
        # Copie indépendante du jeu de données
        data = data.copy()

        # Construction de l'index à partir des colonnes spécifiées
        if time_col is not None or panel_cols is not None:
            # Initialisation de la liste des colonnes à placer en index
            idx_cols = []
            # Ajout des colonnes de panel si spécifiées
            if panel_cols is not None:
                idx_cols.extend(panel_cols)
            # Ajout de la colonne de temps si spécifiées
            if time_col is not None:
                idx_cols.append(time_col)
            # Création de l'index
            if idx_cols:
                data = data.set_index(idx_cols)

        return data

    # Méthode auxiliaire de détection de la régularité d'une série temporelle
    @staticmethod
    def _is_regular_ts(index: pd.DatetimeIndex) -> bool:
        """Check regularity of a single DatetimeIndex.

        Args:
            index: DatetimeIndex to check.

        Returns:
            True if ``pd.infer_freq`` succeeds.
        """
        # Vérification qu'il y a au moins deux observations pour déterminer la fréquence
        if len(index) < 2:
            return False
        # Inférence de la fréquence avec pandas (possible que lorsque l'index est régulier)
        try:
            return pd.infer_freq(index) is not None
        except (TypeError, ValueError):
            return False

    # Méthode auxiliaire de vérification de la régularité d'un panel
    def _is_regular_panel(
        self,
        index: pd.MultiIndex,
        per_entity: bool,
    ) -> Union[bool, Dict[tuple, bool]]:
        """Check regularity for panel data (MultiIndex).

        Args:
            index: MultiIndex of the panel data.
            per_entity: Return per-entity dict or global bool.

        Returns:
            Dict or bool.
        """
        # Extraction du niveau lié à la date
        date_level = index.nlevels - 1
        # Extraction des entités
        entity_index = index.droplevel(date_level)

        # Initialisation du dictionnaire résultat
        results: Dict[tuple, bool] = {}
        # Initialisation de la collection des fréquences détectées
        detected_freqs = set()

        # Parcours des entités
        for entity_key in entity_index.unique():
            # Masque correspondant aux obserbations de l'entité
            mask = entity_index == entity_key
            # Extraction des dates de l'entité
            dates = index.get_level_values(date_level)[mask]
            
            # Vérification de la régularité des dates
            regular = self._is_regular_ts(dates)

            # Construction de la clé associée à l'entité et complétion des résultats
            # Clé TOUJOURS normalisée en tuple, y compris à un seul niveau (§5.4)
            results[normalize_entity_key(entity_key)] = regular

            # Détection de la fréquence
            if regular:
                detected_freqs.add(pd.infer_freq(dates))

        # Cas où l'on attend des résultats par entité
        if per_entity:
            return results

        # Cas où l'on attend un résultat global
        # On vérifie que toutes les entités sont régulières et possèdent la même fréquence
        all_regular = all(results.values())
        same_freq = len(detected_freqs) <= 1
        return all_regular and same_freq

    # Méthode auxiliaire de détection de la fréquence d'une série temporelle
    def _resolve_frequency_ts(
        self,
        data: Union[pd.Series, pd.DataFrame],
    ) -> Optional[str]:
        """Resolve the pandas frequency string for a time series.

        Auto-detects the frequency using ``detect_index_frequency`` and
        ``detect_frequency`` with ``return_format='full'``.

        Args:
            data: Time series data.

        Returns:
            Pandas-compatible frequency string, or None.
        """
        # Détection via detect_index_frequency
        try:
            return detect_index_frequency(data.index, return_format='full')
        except (ValueError, TypeError):
            pass

        # Fallback : détection par colonnes via detect_frequency
        try:
            if isinstance(data, pd.DataFrame):
                freq_map = detect_frequency(data, return_format='full')
                if isinstance(freq_map, dict):
                    self._validate_consistent_positions(freq_map)
                    return _get_highest_frequency(freq_map)
            elif isinstance(data, pd.Series):
                freq = detect_frequency(data, return_format='full')
                if freq is not None and not isinstance(freq, dict):
                    return freq
        except (ValueError, TypeError):
            pass

        return None

    # Méthode auxiliaire de validation de l'ensemble des positions
    @staticmethod
    def _validate_consistent_positions(freq_map: dict) -> None:
        """Validate that all frequencies share the same period position.

        Args:
            freq_map: Dictionary mapping keys to full frequency strings.

        Raises:
            ValueError: If mixed positions (start/end) are detected.
        """
        if not freq_map:
            return

        # Extraction des positions via normalize_frequency
        # Initialisation du dictionnaire des positions
        positions = {}
        # Parcours des fréquences
        for key, freq_str in freq_map.items():
            try:
                # Extraction des positions
                _, position, _ = normalize_frequency(freq_str, return_format='components')
                # Ajout au dictionnaire
                if position is not None:
                    positions[key] = position
            except (ValueError, TypeError):
                continue
        
        # Unicisation des positions
        unique_positions = set(positions.values())
        # Erreur si les positions ne sont pas uniques
        if len(unique_positions) > 1:
            raise ValueError(
                f"Mixed positions detected: {positions}. "
                f"All series must use the same position (start or end)."
            )

    # Méthode de régularisation d'une série temporelle
    def _regularize_ts(
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
        # Création de l'index à la nouvelle fréquence
        new_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=target_frequency,
        )
        # Ajout du nom de l'index
        new_index.name = data.index.name
        # Réindexation des données
        return data.reindex(new_index)

    # Méthode auxiliaire de régularisation de données de panel
    def _regularize_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        per_entity: bool,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize panel data entity by entity.

        Args:
            data: Panel data with MultiIndex.
            per_entity: If True, detect frequency per entity; if False, use a
                single global frequency for all entities.

        Returns:
            Regularized panel data.
        """
        # Extraction du niveau des dates
        date_level = data.index.nlevels - 1
        # Extraction des entités
        entity_index = data.index.droplevel(date_level)
        entity_keys = entity_index.unique()

        # Fréquence globale (si per_entity=False)
        global_freq = None
        if not per_entity:
            # Détection de la fréquence la plus haute sur l'ensemble du panel
            try:
                # Détection de la fréquence
                parsed = detect_index_frequency(data.index, return_format='full')
                if isinstance(parsed, dict):
                    # Unicisation de la position
                    self._validate_consistent_positions(parsed)
                    # Extraction de la fréquence la plus élevée
                    global_freq = _get_highest_frequency(parsed)
                else:
                    global_freq = parsed
            except (ValueError, TypeError):
                pass

            # Fallback : détection par entité puis fréquence la plus haute
            if global_freq is None:
                # Initialisation du dictionnaire des fréquences par entité
                entity_freqs = {}
                # Parcours des entités
                for entity_key in entity_keys:
                    # Création du masque de l'index correspondant à l'entité
                    mask = entity_index == entity_key
                    # Extraction des dates de l'entité
                    dates = data.index.get_level_values(date_level)[mask]
                    # Extraction des données de l'entité
                    entity_data_tmp = data[mask]
                    # Construction de la série temporelle de l'entité
                    if isinstance(entity_data_tmp, pd.DataFrame):
                        entity_data_tmp = entity_data_tmp.copy()
                        entity_data_tmp.index = dates
                    else:
                        entity_data_tmp = pd.Series(entity_data_tmp.values, index=dates)
                    # Détection de la série temporelle de l'entité
                    entity_freq = self._resolve_frequency_ts(entity_data_tmp)
                    # Ajout au dictionnaire
                    if entity_freq is not None:
                        entity_freqs[normalize_entity_key(entity_key)] = entity_freq
                # Extraction de la fréquence la plus élevée
                if entity_freqs:
                    self._validate_consistent_positions(entity_freqs)
                    global_freq = _get_highest_frequency(entity_freqs)

        # Régularisation par entité
        parts = []
        level_names = data.index.names

        # Parcours par entité
        for entity_key in entity_keys:
            # Création du masque par entité
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
                freq = self._resolve_frequency_ts(entity_data)
            else:
                freq = global_freq

            if freq is None:
                # Pas de fréquence détectable → garder tel quel
                parts.append(subset)
                continue

            # Régularisation
            regularized = self._regularize_ts(entity_data, freq)

            # Reconstruction du MultiIndex
            key = normalize_entity_key(entity_key)
            n = len(regularized)

            # Niveaux d'entité
            entity_levels = [np.full(n, k) for k in key]
            arrays = entity_levels + [regularized.index]

            new_mi = pd.MultiIndex.from_arrays(arrays, names=level_names)

            # Attribution de l'index
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

# Fonction indiquant si un index est régulier
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

# Fonction de régularisation d'un index
def regularize(
    data: Union[pd.Series, pd.DataFrame],
    time_col: Optional[str] = None,
    panel_cols: Optional[List[str]] = None,
    per_entity: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Regularize the temporal index by filling gaps with NaN rows.

    Convenience wrapper around :meth:`IndexRegularizer.regularize`.

    Args:
        data: Time series or panel data.
        time_col: Column containing timestamps.
        panel_cols: Panel entity columns.
        per_entity: Detect frequency per entity (panel only).

    Returns:
        Regularized data.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.to_datetime(['2023-01-01', '2023-03-01'])
        >>> result = regularize(pd.Series([1, 3], index=dates))
        >>> len(result)
        3
    """
    return _regularizer.regularize(data, time_col, panel_cols, per_entity)
