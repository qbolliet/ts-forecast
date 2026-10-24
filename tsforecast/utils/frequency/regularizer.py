"""Index regularization utilities for time series data.

This module provides the IndexRegularizer class and utility functions to detect
and fix irregular time series indices (e.g., gaps) by reindexing onto a regular
date_range. The output grid always has a constant frequency: when none can be
built, an error is raised rather than returning an irregular index.
"""
# Importation des modules
import warnings
import pandas as pd
from typing import Dict, List, Optional, Union

# Import des utilitaires internes
from ..utils.frequency.utils import (
    _get_highest_frequency,
    detect_index_frequency,
    normalize_frequency,
    to_pandas_freq,
)
from ..panel.utils import (
    is_panel_data,
    build_panel_index,
    iter_entity_blocks,
)

# Nombre maximal de dates hors grille citées dans un avertissement
_MAX_LISTED_DATES = 5


# Classe de régularisation d'index temporel
class IndexRegularizer:
    """Detect and fix irregular time series indices.

    Provides methods to check whether a DatetimeIndex (or MultiIndex with dates)
    is regular, and to regularize it by filling temporal gaps with NaN rows.

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
    def __init__(self):
        """Initialize the IndexRegularizer.
        """

    #  Méthode de détection de la régularité
    def is_regular(
        self,
        data: Union[pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        per_entity: bool = False,
    ) -> Union[bool, Dict[tuple, bool]]:
        """Check whether the temporal index is regular (no gaps).

        A regular index is one where ``pd.infer_freq`` returns a non-None value
        once the dates are sorted: the row order is not a regularity criterion.
        Fewer than three dates, or a duplicated date, are never regular.

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

        Raises:
            TypeError: If the temporal index is neither a ``DatetimeIndex`` nor
                a ``PeriodIndex``.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='MS')
            >>> series = pd.Series(range(5), index=dates)
            >>> IndexRegularizer().is_regular(series)
            True
        """
        # Préparation des données : copie et mise en index si nécessaire
        data = self._prepare_data(data, time_col, panel_cols)

        # Index de périodes ramené aux dates de début de période, puis vérification du type
        data, _ = self._periods_to_timestamps(data)
        self._check_datetime_index(data.index)

        # Série temporelle simple (DatetimeIndex)
        if not is_panel_data(data):
            return self._infer_freq(data.index) is not None

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

        A ``PeriodIndex`` is regularized on its start-of-period dates and
        returned as a ``PeriodIndex`` of the same frequency. The output grid
        always has a constant frequency. For panel data each
        entity keeps its own date range (min/max); only internal gaps are
        filled. An entity (or series) with fewer than two dates is returned
        as is.

        Observations whose date is not on the detected grid cannot be kept on
        a constant-frequency grid: they are dropped and a ``UserWarning`` lists
        them. If no observation at all falls on the grid, or if no constant
        frequency can be detected, a ``ValueError`` is raised instead.

        Args:
            data: Time series or panel data to regularize.
            time_col: Column containing timestamps. If provided, it is set as
                index before processing and restored as a column afterwards.
            panel_cols: Columns identifying panel entities. If provided, they
                are set as index levels before processing and restored as
                columns afterwards.
            per_entity: For panel data — if True, the target frequency is
                detected independently per entity; if False, a single global
                frequency (the highest one) is used for every entity, so an
                entity of lower frequency is rewritten on that grid.

        Returns:
            Regularized data with the same type as the input.

        Raises:
            TypeError: If the temporal index is neither a ``DatetimeIndex`` nor
                a ``PeriodIndex``.
            ValueError: If duplicated timestamps are found, if no constant
                frequency can be detected, if start and end positions are mixed
                between entities (global mode), or if no observation falls on
                the detected grid.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-04-01'])
            >>> series = pd.Series([1, 2, 4], index=dates)
            >>> result = IndexRegularizer().regularize(series)
            >>> len(result)
            4
        """
        # Préparation : copie de travail + mise en index si nécessaire
        data = self._prepare_data(data, time_col, panel_cols)

        # Index de périodes ramené aux dates de début de période, puis vérification du type
        data, period_freq = self._periods_to_timestamps(data)
        self._check_datetime_index(data.index)

        # Vérification des doublons
        if data.index.duplicated().any():
            raise ValueError("Duplicated timestamps found in the index.")

        # Tri de l'index
        data = data.sort_index()

        # Cas d'une série temporelle
        if not is_panel_data(data):
            # Série temporelle simple : au moins deux dates pour parler de grille
            if len(data) < 2:
                result = data
            else:
                freq = self._detect_frequency(data.index, "The series")
                result = self._regularize_ts(data, freq, "The series")
        else:
            # Panel
            result = self._regularize_panel(data, per_entity)

        # Retour aux périodes d'origine
        if period_freq is not None:
            result = self._timestamps_to_periods(result, period_freq)

        # Restitution en colonnes des niveaux posés en index par _prepare_data
        restored = list(panel_cols or []) + ([time_col] if time_col is not None else [])
        if restored:
            result = result.reset_index(level=restored)

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
            panel_cols: Panel columns to include as index levels. Without
                ``time_col`` they are prepended to the existing (temporal)
                index.

        Returns:
            Copy of data with DatetimeIndex or MultiIndex(entities..., date).
        """
        # Copie indépendante du jeu de données
        data = data.copy()

        # Cas des colonnes de panel seules : l'index existant porte le temps et passe en dernier niveau
        if panel_cols and time_col is None:
            n_levels = data.index.nlevels
            data = data.set_index(list(panel_cols), append=True)
            return data.reorder_levels(
                list(range(n_levels, n_levels + len(panel_cols))) + list(range(n_levels))
            )

        # Construction de l'index à partir des colonnes spécifiées (entités puis temps)
        idx_cols = list(panel_cols or []) + ([time_col] if time_col is not None else [])
        if idx_cols:
            data = data.set_index(idx_cols)

        return data

    # Méthode auxiliaire de conversion d'un index de périodes en dates
    @staticmethod
    def _periods_to_timestamps(data):
        """Convert a ``PeriodIndex`` time level to the start-of-period timestamps.

        Args:
            data: Data whose index is a DatetimeIndex, a PeriodIndex or a
                MultiIndex whose last level is the time.

        Returns:
            Tuple ``(data, period_freq)``: the data (unchanged when its time
            level is not a ``PeriodIndex``) and the original period frequency,
            or None when no conversion took place.
        """
        # Extraction de l'index
        index = data.index
        # Extraction de la valeur temporelle (dernier niveau en cas de MultiIndex)
        times = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
        if not isinstance(times, pd.PeriodIndex):
            return data, None

        # Dates de début de période ; les autres niveaux sont conservés
        data = data.copy()
        # Conversion en timestamp
        data.index = IndexRegularizer._replace_time_level(index, times.to_timestamp())
        return data, times.freq

    # Méthode auxiliaire de retour des dates aux périodes
    @staticmethod
    def _timestamps_to_periods(data, period_freq):
        """Convert the time level back to a ``PeriodIndex`` of the given frequency.

        Args:
            data: Regularized data with a DatetimeIndex time level.
            period_freq: Period frequency of the original index.

        Returns:
            Data with a ``PeriodIndex`` time level.
        """
        # Extraction de l'index
        index = data.index
        # Extraction de la valeur temporelle (dernier niveau si MultiIndex)
        times = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
        # Conversion en périodes
        data.index = IndexRegularizer._replace_time_level(index, times.to_period(period_freq))
        return data

    # Méthode auxiliaire de remplacement du niveau temporel d'un index
    @staticmethod
    def _replace_time_level(index: pd.Index, times: pd.Index) -> pd.Index:
        """Replace the time level (last level) of an index, keeping names and entities.

        Args:
            index: DatetimeIndex, PeriodIndex or MultiIndex (time last).
            times: New time labels, same length as ``index``.

        Returns:
            Index of the same structure with the new time level.
        """
        if isinstance(index, pd.MultiIndex):
            arrays = [index.get_level_values(i) for i in range(index.nlevels - 1)] + [times]
            return pd.MultiIndex.from_arrays(arrays, names=index.names)
        return times.rename(index.name)

    # Méthode auxiliaire de vérification du type de l'index temporel
    @staticmethod
    def _check_datetime_index(index: pd.Index) -> None:
        """Check that the temporal level of an index holds datetimes.

        Args:
            index: DatetimeIndex, or MultiIndex whose last level is the time.

        Raises:
            TypeError: If the temporal level is not a ``DatetimeIndex``
                (e.g. an integer index).
        """
        # Extraction de la valeur temporelle (dernier niveau en cas de MultiIndex)
        times = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
        if not isinstance(times, pd.DatetimeIndex):
            raise TypeError(
                f"The temporal index must be a DatetimeIndex or a PeriodIndex, got {type(times).__name__}."
            )

    # Méthode auxiliaire d'inférence de la fréquence pandas d'un index
    @staticmethod
    def _infer_freq(index: pd.DatetimeIndex) -> Optional[str]:
        """Infer the pandas frequency of a DatetimeIndex, whatever its row order.

        Args:
            index: DatetimeIndex to analyze.

        Returns:
            Pandas frequency string, or None when the dates are not on a
            constant-frequency grid (fewer than three dates, gap, duplicate).
        """
        # Pandas exige trois dates pour inférer une fréquence
        if len(index) < 3:
            return None
        # Tri préalable : pd.infer_freq ne trie pas et rejetterait un index complet mais mélangé
        return pd.infer_freq(index.sort_values())

    # Méthode auxiliaire de détection de la fréquence de la grille cible
    @staticmethod
    def _detect_frequency(index: pd.DatetimeIndex, label: str) -> str:
        """Detect the frequency of the grid an index must be regularized onto.

        Args:
            index: DatetimeIndex with at least two dates.
            label: Subject of the error message (``"The series"``, ``"Entity ('FR',)"``).

        Returns:
            Full pandas frequency string.

        Raises:
            ValueError: If no constant frequency can be detected.
        """
        # Détection de la fréquence de l'index
        freq = detect_index_frequency(index, return_format='full')
        # Message d'erreur si la fréquence n'a pu être détectée
        if freq is None:
            raise ValueError(
                f"{label}: no constant frequency can be detected, the index cannot "
                f"be regularized onto a regular grid."
            )
        return freq

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
        # Initialisation du dictionnaire résultat
        results: Dict[tuple, bool] = {}
        # Initialisation de la collection des fréquences détectées
        detected_freqs = set()

        # Parcours des entités via le primitif de panel partagé avec la régularisation
        # (série factice ne portant que l'index, pour réutiliser iter_entity_blocks)
        dummy = pd.Series(index=index, dtype='float64')
        for entity, _, block in iter_entity_blocks(dummy, is_panel=True):
            # Fréquence des dates de l'entité (None si irrégulière)
            freq = self._infer_freq(block.index)

            # Complétion des résultats (clé TOUJOURS normalisée en tuple par iter_entity_blocks)
            results[entity] = freq is not None
            if freq is not None:
                detected_freqs.add(freq)

        # Cas où l'on attend des résultats par entité
        if per_entity:
            return results

        # Cas où l'on attend un résultat global
        # On vérifie que toutes les entités sont régulières et possèdent la même fréquence
        return all(results.values()) and len(detected_freqs) <= 1

    # Méthode auxiliaire de validation de l'ensemble des positions
    @staticmethod
    def _validate_consistent_positions(freq_map: dict) -> None:
        """Validate that all frequencies share the same period position.

        Args:
            freq_map: Dictionary mapping keys to full frequency strings.

        Raises:
            ValueError: If mixed positions (start/end) are detected.
        """
        # Extraction des positions via normalize_frequency (None pour les fréquences sans position)
        positions = {}
        for key, freq_str in freq_map.items():
            position = normalize_frequency(freq_str, return_format='components').position
            if position is not None:
                positions[key] = position

        # Erreur si les positions ne sont pas uniques
        if len(set(positions.values())) > 1:
            raise ValueError(
                f"Mixed positions detected: {positions}. "
                f"All series must use the same position (start or end)."
            )

    # Méthode de régularisation d'une série temporelle
    def _regularize_ts(
        self,
        data: Union[pd.Series, pd.DataFrame],
        target_frequency: str,
        label: str,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize a single time series (no panel dimension).

        Builds a regular ``date_range`` from min to max of the existing index
        and reindexes the data. Observations that are not on this grid are
        dropped with a warning.

        Args:
            data: Time series data with sorted DatetimeIndex.
            target_frequency: Pandas frequency string (e.g. ``'MS'``, ``'QE-DEC'``).
            label: Subject of the warning / error messages.

        Returns:
            Reindexed data with NaN for filled gaps.

        Raises:
            ValueError: If no observation falls on the grid.
        """
        # Création de l'index à la nouvelle fréquence
        new_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=to_pandas_freq(target_frequency),
        )
        # Ajout du nom de l'index
        new_index.name = data.index.name

        # Repérage des observations hors grille : le reindex les supprimerait en silence
        off_grid = data.index[~data.index.isin(new_index)]
        if len(off_grid) == len(data):
            raise ValueError(
                f"{label}: none of the {len(data)} timestamps falls on the "
                f"'{target_frequency}' grid, the index cannot be regularized."
            )
        if len(off_grid):
            listed = ", ".join(str(d.date()) if d == d.normalize() else str(d)
                               for d in off_grid[:_MAX_LISTED_DATES])
            more = f" (+{len(off_grid) - _MAX_LISTED_DATES} more)" if len(off_grid) > _MAX_LISTED_DATES else ""
            warnings.warn(
                f"{label}: {len(off_grid)} observation(s) not on the '{target_frequency}' "
                f"grid dropped: {listed}{more}.",
                UserWarning,
                stacklevel=3,
            )

        # Réindexation des données
        return data.reindex(new_index)

    # Méthode auxiliaire de régularisation de données de panel
    def _regularize_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        per_entity: bool,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Regularize panel data entity by entity.

        Entity blocks are extracted and re-assembled with the shared panel
        primitives (:func:`iter_entity_blocks`, :func:`build_panel_index`), so
        each entity is regularized by the very same code path as a plain time
        series. Entities with fewer than two dates are kept as they are.

        Args:
            data: Sorted panel data with MultiIndex.
            per_entity: If True, detect frequency per entity; if False, use a
                single global frequency for all entities.

        Returns:
            Regularized panel data.

        Raises:
            ValueError: If an entity has no detectable frequency, or if start
                and end positions are mixed in global mode.
        """
        # Blocs correspondant à chaque entité
        blocks = [
            (entity, mask, block)
            for entity, mask, block in iter_entity_blocks(data, is_panel=True)
        ]
        if not blocks:
            return data

        # Fréquences par entité (None pour les entités à moins de deux dates, laissées telles quelles)
        entity_freqs = {
            entity: (self._detect_frequency(block.index, f"Entity {entity}") if len(block) >= 2 else None)
            for entity, _, block in blocks
        } if per_entity else None

        # Fréquence globale : la plus haute des fréquences d'entités
        global_freq = None
        if not per_entity:
            # Initialisation du dictionnaire des fréquences détectées
            detected = {}
            # Détection pr bloc
            for entity, _, block in blocks:
                if len(block) >= 2:
                    detected[entity] = self._detect_frequency(block.index, f"Entity {entity}")
            # Validation des positions et détection de la fréquence la plus élevée
            if detected:
                self._validate_consistent_positions(detected)
                global_freq = _get_highest_frequency(detected)

        # Régularisation par entité
        parts = []
        for entity, mask, block in blocks:
            # Fréquence de l'entité
            freq = entity_freqs[entity] if per_entity else global_freq
            # Entité à une seule date (ou vide) : gardée telle quelle
            if len(block) < 2:
                parts.append(data[mask])
                continue

            # Régularisation, puis reconstruction du MultiIndex de l'entité
            regularized = self._regularize_ts(block, freq, f"Entity {entity}")
            regularized.index = build_panel_index(
                entity, regularized.index, names=data.index.names
            )
            parts.append(regularized)

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

    Raises:
        TypeError: If the temporal index is neither a ``DatetimeIndex`` nor a
            ``PeriodIndex``.

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

    Raises:
        TypeError: If the temporal index is neither a ``DatetimeIndex`` nor a
            ``PeriodIndex``.
        ValueError: If no constant frequency can be built (see
            :meth:`IndexRegularizer.regularize`).

    Examples:
        >>> import pandas as pd
        >>> dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-04-01'])
        >>> result = regularize(pd.Series([1, 2, 4], index=dates))
        >>> len(result)
        4
    """
    return _regularizer.regularize(data, time_col, panel_cols, per_entity)
