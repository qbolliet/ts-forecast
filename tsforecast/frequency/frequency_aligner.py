"""Frequency alignment utilities for mixed frequency imputation.

This module provides the FrequencyAligner class to aggregate and interpolate
data between different frequencies, handling both time series and panel data.

FrequencyAligner builds homogeneous-frequency datasets for the
HighFrequencyImputer and therefore follows index conventions that differ from
the generic :class:`tsforecast.utils.frequency.converter.FrequencyConverter`
(to which it delegates the actual conversions):

- ``_aggregate_to_target`` preserves the original (dense) index: aggregated
  values are reindexed on it, with NaN outside period boundaries and NaN for
  incomplete periods (``full_periods_only`` semantics).
- ``_interpolate_to_target`` densifies the original index (union with the
  interpolated dates), which covers the **full source periods** of the first
  and last observations: a quarterly series in position start ending on
  2023-07-01 densifies up to 2023-09-01, the last month its Q3 observation
  covers.
- Source frequencies are detected from each variable's observed values (not
  from the index), so a quarterly variable carried on a monthly index is
  handled as quarterly.
"""
# Modules de base
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import normalize_frequency, is_higher_frequency
from ..panel.utils import (
    build_panel_index,
    extract_column_names,
    get_entity_target_frequency,
    group_keys_by_entity_and_variable,
    is_panel_data,
    iter_entity_blocks,
    split_variable_key,
)

# Détection de la fréquence de l'index
from ..utils.frequency.utils import detect_frequency, detect_dataset_frequency, target_offset_for_index


# Classe d'alignement des fréquences de jeu de données avec des fréquences cibles
class FrequencyAligner:
    """Align data to target frequencies via aggregation or interpolation.

    Handles both time series and panel data, delegating actual frequency
    conversion to ``FrequencyConverter``. Panel data is processed per entity
    to respect entity-specific target frequencies.

    Examples:
        >>> import pandas as pd
        >>> aligner = FrequencyAligner()
        >>> dates = pd.date_range('2023-01-01', periods=90, freq='D')
        >>> df = pd.DataFrame({'daily_var': range(90)}, index=dates)
        >>> agg = aligner._aggregate_to_target(
        ...     df, ['daily_var'], 'M'
        ... )
    """

    # Initialisation
    def __init__(self):
        """Initialize the FrequencyAligner."""
        # Initialisation du convertisseur de fréquences
        self._freq_converter = FrequencyConverter()

    # -------------------------------------------------------------------------
    # Itération unifiée séries temporelles / panel
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'itération sur les séries à convertir
    def _iter_variable_series(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> Iterator[Tuple[tuple, Union[np.ndarray, slice], str, pd.Series, str]]:
        """Yield each variable to convert as a date-indexed series.

        Single entry point of the time series / panel disjunction: a time
        series is iterated as a panel holding one entity ``()`` covering
        every row, so aggregation and interpolation are written once, against
        a plain date-indexed series.

        Args:
            df: Input DataFrame.
            keys: Variable keys to convert (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Yields:
            Tuple ``(entity, mask, col, series, entity_target)`` where
            ``mask`` selects the entity's rows in ``df``, ``series`` is the
            column restricted to the entity and indexed by date only, and
            ``entity_target`` the target frequency resolved for the entity.
        """
        # Regroupement des colonnes par entité : hors panel, l'entité portée par
        # une éventuelle clé-tuple est ignorée (une seule série par colonne)
        grouped = (
            group_keys_by_entity_and_variable(keys) if is_panel
            else {(): extract_column_names(keys)}
        )

        # Parcours des blocs d'entités (bloc unique hors panel)
        for entity, mask, block in iter_entity_blocks(df, is_panel=is_panel):
            # Entité sans colonne à convertir
            cols = grouped.get(entity)
            if not cols:
                continue

            # Fréquence cible de l'entité : l'entité vide d'une série temporelle
            # ne peut être servie que par une cible globale
            entity_target = get_entity_target_frequency(entity, target_frequency)

            # Parcours des colonnes de l'entité
            for col in cols:
                # Colonne absente du jeu de données : ignorée
                if col not in df.columns:
                    continue
                yield entity, mask, col, block[col], entity_target

    # Méthode auxiliaire d'affectation d'une colonne sur les lignes d'une entité
    @staticmethod
    def _assign_entity_values(
        result: pd.DataFrame,
        mask: Union[np.ndarray, slice],
        col: str,
        values: np.ndarray,
    ) -> None:
        """Assign values to the rows of one entity, in place.

        Args:
            result: Frame to update in place.
            mask: Boolean array selecting the entity's rows.
            col: Column to update.
            values: Values to write, positionally aligned on ``mask``.
        """
        # Promotion en flottant des colonnes entières ou booléennes : les valeurs
        # converties portent des NaN (hors bornes de période, périodes
        # incomplètes) qu'un dtype entier ne peut pas accueillir
        if result[col].dtype.kind in 'iub':
            result[col] = result[col].astype(float)

        # Affectation positionnelle sur les lignes de l'entité
        result.loc[mask, col] = values

    # -------------------------------------------------------------------------
    # Conversions unitaires, sur une série indexée par dates
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de sélection de la série à resampler
    @staticmethod
    def _observed_series(series: pd.Series) -> pd.Series:
        """Select the series to resample: observed values when they are regular.

        Aggregating the observed values only (instead of the full series) lets
        the converter detect the variable's own frequency rather than the
        index frequency — decisive for sparse variables (a quarterly variable
        carried on a monthly index would otherwise be seen as a monthly series
        with two thirds of its observations missing, and every period would be
        masked by ``full_periods_only``).

        The restriction is applied only when the observed dates form a regular
        index: otherwise the variable has holes of its own and the index
        frequency remains the only usable reference.

        Args:
            series: Column to aggregate, possibly sparse.

        Returns:
            The observed values when they are regularly spaced, the original
            series otherwise.
        """
        # Restriction aux valeurs observées
        observed = series.dropna()
        # Série vide : aucune restriction possible
        if observed.empty:
            return series
        # Vérification de la régularité de l'index des observations
        try:
            inferred = observed.index.inferred_freq
        except (ValueError, TypeError):
            inferred = None
        # Repli sur la série complète si les observations sont irrégulières
        return observed if inferred is not None else series

    # Méthode auxiliaire d'agrégation d'une série indexée par dates
    def _aggregate_series(
        self,
        series: pd.Series,
        target_frequency: str,
    ) -> Optional[pd.Series]:
        """Aggregate one date-indexed series onto its own index.

        Args:
            series: Column to aggregate, indexed by date only.
            target_frequency: Target frequency of the series.

        Returns:
            Aggregated series reindexed on ``series``'s index (NaN outside
            period boundaries and for incomplete periods), or None when the
            series holds no observation at all.
        """
        # Offset ancré comme l'index source, pour que les labels agrégés
        # retombent sur des dates présentes dans l'index d'origine
        target_offset = target_offset_for_index(series.index, target_frequency)

        # Restriction aux valeurs observées : la fréquence source doit être
        # celle de la variable, pas celle de l'index
        observed = self._observed_series(series)

        # Série sans aucune observation : rien à agréger
        if observed.dropna().empty:
            return None

        # Agrégation à la fréquence cible
        aggregated = self._freq_converter.aggregate_to_lower_frequency(
            observed, target_offset, method='sum', full_periods_only=True
        )

        # Reproduction de l'index d'origine
        return aggregated.reindex(series.index)

    # Méthode auxiliaire d'interpolation d'une série indexée par dates
    def _interpolate_series(
        self,
        series: pd.Series,
        target_frequency: str,
        method: str = 'linear',
        limit: Union[int, Literal['default'], None] = 'default',
        limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.Series:
        """Interpolate one date-indexed series to a higher frequency.

        Args:
            series: Column to interpolate, indexed by date only.
            target_frequency: Target frequency of the series.
            method: Interpolation method passed to the converter.
            limit: Maximum number of consecutive NaN values to fill.
            limit_direction: Direction in which to fill NaN values.
            limit_area: Restriction area for filling NaN values.

        Returns:
            Interpolated series, indexed on the target frequency grid covering
            the full source periods of the first and last observations.
        """
        # Interpolation à la fréquence cible : la fréquence source est détectée
        # sur les valeurs observées de la variable (et non sur l'index) pour
        # respecter la fréquence propre de la variable

        # Restriction aux valeurs observées : la fréquence source doit être
        # celle de la variable, pas celle de l'index
        observed = self._observed_series(series)

        # Série sans aucune observation : rien à agréger
        if observed.dropna().empty:
            return None
        
        return self._freq_converter.interpolate_to_higher_frequency(
            observed,
            target_frequency,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            source_freq=detect_frequency(series, return_format='with_position'),
        )

    # -------------------------------------------------------------------------
    # Conversions de jeux de données
    # -------------------------------------------------------------------------
    # Méthode d'aggrégation des données à une fréquence cible
    def _aggregate_to_target(
        self,
        df: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        For panel data, aggregation is performed per entity to respect
        entity-specific target frequencies.

        Contract: this method never changes the number or order of rows —
        the output index is always identical to ``df``'s. Aggregated values
        are reindexed onto it (NaN outside period boundaries and for
        incomplete periods); callers that want to compact the result must do
        so explicitly.

        The source frequency is detected from each variable's **observed
        values**, not from the index: only the non-NaN observations are
        resampled, so a quarterly variable carried on a monthly index is
        aggregated as quarterly (4 sub-periods per year) instead of being
        treated as an incomplete monthly series (which would mask every
        period as NaN under ``full_periods_only``).

        Args:
            df: Input DataFrame.
            aggregate_keys: Variable keys to aggregate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).

        Returns:
            DataFrame with aggregated columns, indexed exactly like ``df``.
        """
        # Cas où les clés d'aggrégation ne sont pas spécifiées
        if not aggregate_keys:
            return df

        # Initialisation du jeu de données résultat
        result = df.copy()

        # Parcours des séries à agréger (séries temporelles et panel confondus)
        for _, mask, col, series, entity_target in self._iter_variable_series(
            df, aggregate_keys, target_frequency, is_panel_data(df)
        ):
            # Agrégation de la série sur son propre index
            aggregated = self._aggregate_series(series, entity_target)
            # Colonne sans observation pour l'entité : laissée telle quelle
            if aggregated is None:
                continue
            # Affectation aux lignes de l'entité
            self._assign_entity_values(result, mask, col, aggregated.values)

        return result

    # Méthode auxiliaire de construction de l'index densifié
    def build_densified_index(
        self,
        df: pd.DataFrame,
        interpolated: Dict[tuple, Dict[str, pd.Series]],
        is_panel: bool,
    ) -> pd.Index:
        """Build the index densified with the interpolated dates.

        The time index of each entity is the union of its original dates and
        the dates produced by interpolation, so that entity-specific target
        frequencies are respected. The union is kept whole: interpolation
        covers the full source periods of the first and last observations,
        and those sub-periods legitimately belong to the data (a quarterly
        observation in position start covers the two months that follow it,
        in position end the two months that precede it).

        Args:
            df: Original DataFrame (DatetimeIndex or panel MultiIndex).
            interpolated: Interpolated series, keyed by entity then column.
            is_panel: Whether the data is panel data.

        Returns:
            DatetimeIndex for a time series, MultiIndex combining each entity
            with its densified time index for a panel.
        """
        # Initialisation des morceaux d'index par entité
        parts: List[pd.Index] = []

        # Parcours des blocs d'entités (bloc unique hors panel)
        for entity, _, block in iter_entity_blocks(df, is_panel=is_panel):
            # Union des dates d'origine et des dates issues de l'interpolation
            densified_times = block.index
            for series in interpolated.get(entity, {}).values():
                densified_times = densified_times.union(series.index)

            # Reconstruction du MultiIndex de l'entité pour un panel
            parts.append(
                build_panel_index(entity, densified_times, names=df.index.names)
                if is_panel else densified_times
            )

        # Panel sans aucune entité : index d'origine conservé
        if not parts:
            return df.index

        # Concaténation des morceaux, entité par entité
        return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]

    # Méthode d'interpolation d'un jeu de données à une fréquence cible
    def _interpolate_to_target(
        self,
        df: pd.DataFrame,
        interpolate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        method: str = 'linear',
        limit: Union[int, Literal['default'], None] = 'default',
        limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        The resulting index is **densified**: it is the union of the original
        index and the dates generated by interpolation. Passing a sparse frame
        (e.g. quarterly observations only) therefore yields a frame at the
        target frequency, while passing a frame already indexed at the target
        frequency simply fills its NaN holes. Columns that are not interpolated
        keep their original values and are NaN on the newly created dates.

        Densification spans the **full source periods** of the first and last
        observations, whose sub-periods are covered by the data: a ``QS``
        series ending on 2023-07-01 densifies to ``MS`` up to 2023-09-01, a
        ``QE`` series starting on 2023-03-31 densifies to ``ME`` down to
        2023-01-31. The output index is therefore generally wider than the
        input one — callers that must stay on their own grid have to reindex
        explicitly.

        For panel data, interpolation and densification are performed per
        entity to respect entity-specific target frequencies.

        The source frequency is detected from each variable's observed values,
        not from the index, so a quarterly variable carried on a monthly index
        is correctly recognised as quarterly.

        Args:
            df: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            method: Interpolation method passed to
                :meth:`FrequencyConverter.interpolate_to_higher_frequency`
                (e.g. ``'linear'``, ``'time'``). Defaults to ``'linear'``.
            limit: Maximum number of consecutive NaN values to fill. If
                ``'default'``, computed automatically as the frequency conversion
                factor between the variable's own frequency and the target
                frequency (e.g. 3 for quarterly→monthly). Pass an explicit
                integer to override this behaviour, or None for no limit.
            limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            limit_area: Restriction area for filling NaN values
                (``'inside'`` or ``'outside'``). Passed directly to
                :meth:`FrequencyConverter.interpolate_to_higher_frequency`.

        Returns:
            DataFrame with interpolated columns, indexed on the densified index.

        Examples:
            >>> import pandas as pd
            >>> aligner = FrequencyAligner()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='QS')
            >>> df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)
            >>> out = aligner._interpolate_to_target(
            ...     df, ['gdp'], 'MS'
            ... )
            >>> len(out)
            9
        """
        # Cas où aucune clé d'interpolation n'est spécifiée, retourne le jeu de données inchangé
        if not interpolate_keys:
            return df

        # Détection unique de la structure du jeu de données (série ou panel)
        is_panel = is_panel_data(df)

        # Interpolation de chaque série (séries temporelles et panel confondus)
        interpolated: Dict[tuple, Dict[str, pd.Series]] = {}
        for entity, _, col, series, entity_target in self._iter_variable_series(
            df, interpolate_keys, target_frequency, is_panel
        ):
            interpolated.setdefault(entity, {})[col] = self._interpolate_series(
                series,
                entity_target,
                method=method,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )

        # Densification de l'index, entité par entité
        target_index = self.build_densified_index(df, interpolated, is_panel)

        # Réindexation du jeu de données sur l'index densifié
        result = df.reindex(target_index)

        # Affectation des colonnes interpolées, entité par entité
        for entity, mask, block in iter_entity_blocks(result, is_panel=is_panel):
            # Entité sans colonne interpolée
            cols = interpolated.get(entity)
            if not cols:
                continue

            # Parcours des colonnes interpolées
            for col, series in cols.items():
                # Les valeurs interpolées priment, complétées par les observations
                # d'origine situées hors de la grille cible (positions de période
                # différentes)
                filled = series.combine_first(block[col]).reindex(block.index)
                # Affectation aux lignes de l'entité
                self._assign_entity_values(result, mask, col, filled.values)

        return result

    # Méthode générique de conversion vers la fréquence cible (agrégation ou interpolation)
    def convert_to_target(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        interp_method: str = 'linear',
        interp_limit: Union[int, Literal['default'], None] = 'default',
        interp_limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        interp_limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Convert columns to a target frequency, aggregating or interpolating as needed.

        For each variable key (or per entity for panel data), the source
        frequency is detected from the variable's observed values. If the
        target frequency is **higher** (more granular) than the source, the
        column is interpolated via :meth:`_interpolate_to_target`. Otherwise it
        is aggregated via :meth:`_aggregate_to_target`.

        Args:
            df: Input DataFrame.
            keys: Variable keys to convert (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            interp_method: Interpolation method used when upsampling.
                Defaults to ``'linear'``.
            interp_limit: Maximum number of consecutive NaN values to fill during
                interpolation. If ``'default'``, computed automatically from the
                frequency conversion factor. See
                :meth:`_interpolate_to_target` for details.
            interp_limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            interp_limit_area: Restriction area for NaN filling during
                interpolation (``'inside'`` or ``'outside'``). Forwarded to
                :meth:`_interpolate_to_target`.

        Returns:
            DataFrame with converted columns.
        """
        # Cas où aucune clé n'est spécifiée
        if not keys:
            return df

        # Détection unique de la structure du jeu de données (série ou panel)
        is_panel = is_panel_data(df)

        # Initialisation des listes de clés selon la direction de conversion
        aggregate_keys: List[Union[str, Tuple]] = []
        interpolate_keys: List[Union[str, Tuple]] = []

        # Détection des fréquences sources (col → freq pour TS, (entité, col) → freq pour panel)
        freq_map = detect_dataset_frequency(df)

        # Classification des clés selon la relation fréquence source / fréquence cible
        for key in keys:
            # Décomposition de la clé : entité toujours en tuple, () hors panel
            entity, col = split_variable_key(key)
            # Vérification que la colonne est dans le jeu de données
            if col not in df.columns:
                continue
            # Clé de lookup : nom de colonne pour TS, tuple complet pour panel
            lookup_key = key if is_panel and isinstance(key, tuple) else col
            # Extraction de la fréquence source
            source_freq = freq_map.get(lookup_key)
            # Fréquence cible : globale pour TS, par entité pour panel
            key_target = get_entity_target_frequency(
                entity if is_panel else (), target_frequency
            )
            # Orientation de la conversion
            if source_freq and is_higher_frequency(
                normalize_frequency(key_target),
                normalize_frequency(source_freq),
            ):
                interpolate_keys.append(key)
            else:
                aggregate_keys.append(key)

        # Application des conversions
        result = df
        if aggregate_keys:
            result = self._aggregate_to_target(
                result, aggregate_keys, target_frequency
            )
        if interpolate_keys:
            result = self._interpolate_to_target(
                result, interpolate_keys, target_frequency,
                method=interp_method,
                limit=interp_limit,
                limit_direction=interp_limit_direction,
                limit_area=interp_limit_area,
            )
        return result
