"""Frequency alignment utilities for mixed frequency imputation.

This module provides the FrequencyAligner class to aggregate and interpolate
data between different frequencies, handling both time series and panel data.

FrequencyAligner builds homogeneous-frequency datasets for the
HighFrequencyImputer and therefore follows index conventions that differ from
the generic :class:`tsforecast.utils.frequency.converter.FrequencyConverter`
(to which it delegates the actual conversions):

- ``aggregate_to_target`` preserves the original (dense) index: aggregated
  values are reindexed on it, with NaN outside period boundaries and NaN for
  incomplete periods (``full_periods_only`` semantics).
- ``interpolate_to_target`` densifies the original index (union with the
  interpolated dates, restricted to the original time span).
- Source frequencies are detected from each variable's observed values (not
  from the index), so a quarterly variable carried on a monthly index is
  handled as quarterly.
"""
# Modules de base
from typing import Dict, List, Literal, Optional, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import normalize_frequency, is_higher_frequency
from ..panel.utils import (
    get_unique_panel_entities,
    group_keys_by_entity_and_variable,
    extract_column_names,
    get_entity_mask,
    get_entity_target_frequency,
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
        >>> agg = aligner.aggregate_to_target(
        ...     df, ['daily_var'], 'M', is_panel=False
        ... )
    """

    # Initialisation
    def __init__(self):
        """Initialize the FrequencyAligner."""
        # Initialisation du convertisseur de fréquences
        self._freq_converter = FrequencyConverter()

    # Méthode de regroupement des variables par entité
    # Délégation à la fonction utilitaire partagée de tsforecast.panel.utils
    def group_keys_by_entity_and_variable(
        self,
        keys: List[Union[str, Tuple]],
    ) -> Dict[tuple, List[str]]:
        """Group variable keys by entity, extracting column names.

        Thin wrapper over :func:`tsforecast.panel.utils.group_keys_by_entity_and_variable`.

        Args:
            keys: List of (entity..., variable) tuples or plain column names.

        Returns:
            Dict mapping entity tuples to lists of column names.
            For time series (non-panel), returns {(): [col_names]}.
        """
        return group_keys_by_entity_and_variable(keys)

    # Méthode d'extraction de la fréquence cible d'une entité
    # Délégation à la fonction utilitaire partagée de tsforecast.panel.utils
    def get_entity_target_frequency(
        self,
        entity: tuple,
        target_frequency: Union[str, Dict],
    ) -> str:
        """Get the target frequency for a specific entity.

        Thin wrapper over :func:`tsforecast.panel.utils.get_entity_target_frequency`.

        Args:
            entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).
            target_frequency: Either a single frequency string or a dict
                mapping entities to frequencies.

        Returns:
            Target frequency string for the entity.

        Raises:
            ValueError: If no target frequency found for the entity.
        """
        return get_entity_target_frequency(entity, target_frequency)

    # Méthode de construction du masque des observations associées à une entité
    # Délégation à la fonction utilitaire partagée de tsforecast.panel.utils
    def get_entity_mask(
        self,
        X: pd.DataFrame,
        entity: tuple,
    ) -> np.ndarray:
        """Get a boolean mask for rows belonging to a specific entity.

        Thin wrapper over :func:`tsforecast.panel.utils.get_entity_mask`.

        Args:
            X: DataFrame with MultiIndex (entity levels + time level).
            entity: Entity tuple to filter on.

        Returns:
            Boolean numpy array of shape (len(X),).
        """
        return get_entity_mask(X, entity)

    # Méthode auxiliaire de construction d'un offset positionné comme l'index source
    # Délégation à la fonction utilitaire partagée de tsforecast.utils.frequency.utils
    def _target_offset_for_index(
        self,
        index: pd.DatetimeIndex,
        target_frequency: str,
    ) -> str:
        """Build a pandas offset for target_frequency anchored like the index.

        Thin wrapper over :func:`tsforecast.utils.frequency.utils.target_offset_for_index`.

        Args:
            index: Source DatetimeIndex.
            target_frequency: Base target frequency (e.g. 'Q').

        Returns:
            Pandas offset alias (e.g. 'QS' for a month-start index).
        """
        return target_offset_for_index(index, target_frequency)

    # Méthode auxiliaire de sélection de la série à resampler
    @staticmethod
    def _observed_series_for_aggregation(series: pd.Series) -> pd.Series:
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

    # Méthode d'aggrégation des données à une fréquence cible
    def aggregate_to_target(
        self,
        df: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
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
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with aggregated columns, indexed exactly like ``df``.
        """
        # Cas où les clés d'aggrégation ne sont pas spécifiées
        if not aggregate_keys:
            return df

        # Initialisation du jeu de données résultat
        result = df.copy()

        # Cas séries temporelles : agrégation globale
        if not is_panel:
            # Extraction des colonnes des tuples
            columns = self.extract_column_names(aggregate_keys)
            # Offset ancré comme l'index source, pour que les labels agrégés
            # retombent sur des dates présentes dans l'index d'origine
            target_offset = self._target_offset_for_index(df.index, target_frequency)
            # Parcours des colonnes
            for col in columns:
                # Vérification que la colonne est dans le jeu de données
                if col not in df.columns:
                    continue
                # Restriction aux valeurs observées : la fréquence source doit être
                # celle de la variable, pas celle de l'index
                observed = self._observed_series_for_aggregation(df[col])
                # Colonne sans aucune observation : laissée telle quelle
                if observed.dropna().empty:
                    continue
                # Aggrégation à la fréquence cible
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    observed, target_offset, method='sum', full_periods_only=True
                )
                # Reproduction de l'index original
                result[col] = aggregated.reindex(df.index)
            return result

        # Cas panel : agrégation par entité
        # Extraction du dictionnaire associant une liste de variables à chaque entité
        grouped = self.group_keys_by_entity_and_variable(aggregate_keys)

        # Parcours des entités
        for entity, cols in grouped.items():
            # Extraction de la fréquence cible associée à l'enité
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            # Création du masque des observations de l'entité
            entity_mask = self.get_entity_mask(df, entity)

            # Parcours des colonnes
            for col in cols:
                # Cas où la colonne n'est pas présente dans le jeu de données
                if col not in df.columns:
                    continue

                # Extraction des observations de l'entité pour la colonne d'intérêt
                entity_series = df.loc[entity_mask, col]
                # Suppression des niveaux afférents à l'entité
                entity_series = entity_series.droplevel(
                    list(range(df.index.nlevels - 1))
                )
                # Offset ancré comme l'index de l'entité
                target_offset = self._target_offset_for_index(
                    entity_series.index, entity_target
                )
                # Restriction aux valeurs observées : la fréquence source doit être
                # celle de la variable, pas celle de l'index
                observed = self._observed_series_for_aggregation(entity_series)
                # Colonne sans aucune observation pour l'entité : laissée telle quelle
                if observed.dropna().empty:
                    continue
                # Agrégation à la fréquence souhaitée
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    observed, target_offset, method='sum', full_periods_only=True
                )
                # Réindexation à l'index temporel original
                reindexed = aggregated.reindex(entity_series.index)
                # Ajout au DataFrame résultat
                result.loc[entity_mask, col] = reindexed.values

        return result


    # Méthode auxiliaire de restriction d'un index densifié à la plage d'origine
    def restrict_to_original_span(
        self,
        densified_index: pd.DatetimeIndex,
        original_index: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """Restrict a densified index to the time span of the original index.

        Densification adds intermediate dates but must not extend the period
        covered by the data: interpolation can generate dates beyond the last
        observation (up to the end of its source period).

        Args:
            densified_index: Index densified with the interpolated dates.
            original_index: Original time index.

        Returns:
            Densified index restricted to [min, max] of the original index.
        """
        # Cas d'un index original vide : aucune borne à appliquer
        if len(original_index) == 0:
            return densified_index

        # Restriction aux bornes de l'index original
        return densified_index[
            (densified_index >= original_index.min())
            & (densified_index <= original_index.max())
        ]

    # Méthode auxiliaire de construction de l'index densifié d'un panel
    def build_densified_panel_index(
        self,
        df: pd.DataFrame,
        interpolated: Dict[tuple, Dict[str, pd.Series]],
    ) -> pd.MultiIndex:
        """Build a panel MultiIndex densified with the interpolated dates.

        For each entity, the time index is the union of its original dates and
        the dates produced by interpolation, so that entity-specific target
        frequencies are respected.

        Args:
            df: Original panel DataFrame with MultiIndex.
            interpolated: Interpolated series, keyed by entity then column.

        Returns:
            MultiIndex combining each entity with its densified time index.
        """
        # Initialisation des tuples de l'index résultat
        tuples: List[tuple] = []

        # Parcours des entités du panel
        for entity in get_unique_panel_entities(df):
            # Extraction des dates originales de l'entité
            entity_mask = self.get_entity_mask(df, entity)
            entity_times = df.index.get_level_values(-1)[entity_mask]

            # Union avec les dates issues de l'interpolation
            densified_times = entity_times
            for series in interpolated.get(entity, {}).values():
                densified_times = densified_times.union(series.index)

            # Restriction à la plage temporelle d'origine de l'entité
            densified_times = self.restrict_to_original_span(densified_times, entity_times)

            # Construction des tuples (entité..., date)
            tuples.extend((*entity, time) for time in densified_times)

        return pd.MultiIndex.from_tuples(tuples, names=df.index.names)

    # Méthode d'interpolation d'un jeu de données à une fréquence cible
    def interpolate_to_target(
        self,
        df: pd.DataFrame,
        interpolate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
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
            is_panel: Whether the data is panel data.
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
            >>> out = aligner.interpolate_to_target(
            ...     df, ['gdp'], 'MS', is_panel=False
            ... )
            >>> len(out)
            9
        """
        # Cas où aucune clé d'interpolation n'est spécifiée, retourne le jeu de données inchangé
        if not interpolate_keys:
            return df

        # Cas séries temporelles : interpolation globale
        if not is_panel:
            # Extraction des colonnes
            columns = self.extract_column_names(interpolate_keys)

            # Interpolation de chaque colonne
            interpolated: Dict[str, pd.Series] = {}
            for col in columns:
                # Vérification que la colonne est bien dans le jeu de données
                if col not in df.columns:
                    continue

                # Interpolation à la fréquence cible
                interpolated[col] = self._freq_converter.interpolate_to_higher_frequency(
                    df[col],
                    target_frequency,
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction,
                    limit_area=limit_area,
                    source_freq=detect_frequency(df[col], return_format='with_position'),
                )

            # Densification de l'index : union de l'index original et des index interpolés
            target_index = df.index
            for series in interpolated.values():
                target_index = target_index.union(series.index)

            # Restriction à la plage temporelle d'origine
            target_index = self.restrict_to_original_span(target_index, df.index)

            # Réindexation du jeu de données sur l'index densifié
            result = df.reindex(target_index)

            # Affectation des colonnes interpolées
            for col, series in interpolated.items():
                # Les valeurs interpolées priment, complétées par les observations
                # d'origine situées hors de l'index cible (positions de période différentes)
                result[col] = (
                    series.reindex(target_index)
                    .combine_first(df[col].reindex(target_index))
                )

            return result

        # Cas panel : interpolation par entité
        # Regroupement des colonnes par entité
        grouped = self.group_keys_by_entity_and_variable(interpolate_keys)

        # Interpolation des séries de chaque entité
        interpolated_panel: Dict[tuple, Dict[str, pd.Series]] = {}
        for entity, cols in grouped.items():
            # Extraction de la fréquence cible associée à l'entité
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            # Extraction du masque des observations associées à l'entité
            entity_mask = self.get_entity_mask(df, entity)

            # Parcours des colonnes
            for col in cols:
                # Vérification que la colonne est dans le jeu de données
                if col not in df.columns:
                    continue

                # Extraction des observations de la colonne associées à l'entité
                entity_series = df.loc[entity_mask, col]
                # Suppression des niveaux associés à l'entité
                entity_series = entity_series.droplevel(
                    list(range(df.index.nlevels - 1))
                )

                # Interpolation à la fréquence cible de l'entité
                # La fréquence source est détectée sur les valeurs observées de la variable
                # (et non sur l'index) pour respecter la fréquence propre de la variable
                interpolated_panel.setdefault(entity, {})[col] = self._freq_converter.interpolate_to_higher_frequency(
                    entity_series,
                    entity_target,
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction,
                    limit_area=limit_area,
                    source_freq=detect_frequency(entity_series, return_format='with_position'),
                )

        # Densification de l'index du panel entité par entité
        target_index = self.build_densified_panel_index(df, interpolated_panel)
        # Réindexation du jeu de données sur l'index densifié
        result = df.reindex(target_index)
        # Nombre de niveaux d'entité de l'index résultat
        entity_levels = list(range(target_index.nlevels - 1))

        # Affectation des colonnes interpolées par entité
        for entity, cols in interpolated_panel.items():
            # Extraction du masque des observations de l'entité dans l'index densifié
            entity_mask = self.get_entity_mask(result, entity)
            # Extraction des dates de l'entité
            entity_times = target_index.get_level_values(-1)[entity_mask]

            # Parcours des colonnes interpolées
            for col, series in cols.items():
                # Extraction des valeurs originales réindexées de l'entité
                original = result.loc[entity_mask, col].droplevel(entity_levels)
                # Les valeurs interpolées priment, complétées par les observations d'origine
                filled = series.combine_first(original).reindex(entity_times)
                # Affectation au résultat
                result.loc[entity_mask, col] = filled.values

        return result

    # Méthode générique de conversion vers la fréquence cible (agrégation ou interpolation)
    def convert_to_target(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
        interp_method: str = 'linear',
        interp_limit: Union[int, Literal['default'], None] = 'default',
        interp_limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        interp_limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Convert columns to a target frequency, aggregating or interpolating as needed.

        For each variable key (or per entity for panel data), the source
        frequency is detected from the variable's observed values. If the
        target frequency is **higher** (more granular) than the source, the
        column is interpolated via :meth:`interpolate_to_target`. Otherwise it
        is aggregated via :meth:`aggregate_to_target`.

        Args:
            df: Input DataFrame.
            keys: Variable keys to convert (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.
            interp_method: Interpolation method used when upsampling.
                Defaults to ``'linear'``.
            interp_limit: Maximum number of consecutive NaN values to fill during
                interpolation. If ``'default'``, computed automatically from the
                frequency conversion factor. See
                :meth:`interpolate_to_target` for details.
            interp_limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            interp_limit_area: Restriction area for NaN filling during
                interpolation (``'inside'`` or ``'outside'``). Forwarded to
                :meth:`interpolate_to_target`.

        Returns:
            DataFrame with converted columns.
        """
        # Cas où aucune clé n'est spécifiée
        if not keys:
            return df

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
            key_target = self.get_entity_target_frequency(
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
            result = self.aggregate_to_target(
                result, aggregate_keys, target_frequency, is_panel
            )
        if interpolate_keys:
            result = self.interpolate_to_target(
                result, interpolate_keys, target_frequency, is_panel,
                method=interp_method,
                limit=interp_limit,
                limit_direction=interp_limit_direction,
                limit_area=interp_limit_area,
            )
        return result

    # Méthode d'extraction des noms de colonnes
    # Délégation à la fonction utilitaire partagée de tsforecast.panel.utils
    def extract_column_names(
        self,
        keys: List[Union[str, Tuple]],
    ) -> List[str]:
        """Extract unique column names from variable keys.

        Thin wrapper over :func:`tsforecast.panel.utils.extract_column_names`.

        Args:
            keys: List of variable identifiers (column names or
                (entity, column) tuples).

        Returns:
            List of unique column names.
        """
        return extract_column_names(keys)
