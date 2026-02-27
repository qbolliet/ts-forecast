"""Frequency alignment utilities for mixed frequency imputation.

This module provides the FrequencyAligner class to aggregate and interpolate
data between different frequencies, handling both time series and panel data.
"""
# Modules de base
from typing import Dict, List, Optional, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import normalize_frequency
from ..panel.utils import normalize_entity_key


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

    def __init__(self):
        """Initialize the FrequencyAligner."""
        self._freq_converter = FrequencyConverter()

    def group_keys_by_entity_and_variable(
        self,
        keys: List[Union[str, Tuple]],
    ) -> Dict[tuple, List[str]]:
        """Group variable keys by entity, extracting column names.

        Args:
            keys: List of (entity..., variable) tuples or plain column names.

        Returns:
            Dict mapping entity tuples to lists of column names.
            For time series (non-panel), returns {(): [col_names]}.
        """
        grouped: Dict[tuple, List[str]] = {}

        for key in keys:
            if isinstance(key, tuple):
                entity = normalize_entity_key(key[:-1])
                col = key[-1]
            else:
                entity = ()
                col = key

            if entity not in grouped:
                grouped[entity] = []
            if col not in grouped[entity]:
                grouped[entity].append(col)

        return grouped

    def get_entity_target_frequency(
        self,
        entity: tuple,
        target_frequency: Union[str, Dict],
    ) -> str:
        """Get the target frequency for a specific entity.

        Args:
            entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).
            target_frequency: Either a single frequency string or a dict
                mapping entities to frequencies.

        Returns:
            Normalized target frequency string for the entity.

        Raises:
            ValueError: If no target frequency found for the entity.
        """
        if isinstance(target_frequency, dict):
            freq = target_frequency.get(entity)
            if freq is None and len(entity) == 1:
                freq = target_frequency.get(entity[0])
            if freq is None:
                raise ValueError(
                    f"No target frequency found for entity {entity}"
                )
            return freq

        return target_frequency

    def get_entity_mask(
        self,
        X: pd.DataFrame,
        entity: tuple,
    ) -> np.ndarray:
        """Get a boolean mask for rows belonging to a specific entity.

        Args:
            X: DataFrame with MultiIndex (entity levels + time level).
            entity: Entity tuple to filter on.

        Returns:
            Boolean numpy array of shape (len(X),).
        """
        index = X.index
        n_entity_levels = index.nlevels - 1

        if n_entity_levels == 1:
            return (index.get_level_values(0) == entity[0])
        else:
            mask = np.ones(len(X), dtype=bool)
            for i in range(n_entity_levels):
                mask &= (index.get_level_values(i) == entity[i])
            return mask

    def aggregate_to_target(
        self,
        X: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        For panel data, aggregation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            X: Input DataFrame.
            aggregate_keys: Variable keys to aggregate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with aggregated columns.
        """
        if not aggregate_keys:
            return X

        result = X.copy()

        # Cas séries temporelles : agrégation globale
        if not is_panel:
            columns = self.extract_column_names(aggregate_keys)

            for col in columns:
                if col not in X.columns:
                    continue
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    X[col], target_frequency, method='sum'
                )
                result[col] = aggregated.reindex(X.index)

            result.dropna(axis=0, how='all', inplace=True)
            return result

        # Cas panel : agrégation par entité
        grouped = self.group_keys_by_entity_and_variable(aggregate_keys)

        for entity, cols in grouped.items():
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            entity_mask = self.get_entity_mask(X, entity)

            for col in cols:
                if col not in X.columns:
                    continue

                entity_series = X.loc[entity_mask, col]
                entity_series = entity_series.droplevel(
                    list(range(X.index.nlevels - 1))
                )

                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    entity_series, entity_target, method='sum'
                )

                reindexed = aggregated.reindex(entity_series.index)
                result.loc[entity_mask, col] = reindexed.values

        result.dropna(axis=0, how='all', inplace=True)
        return result

    def interpolate_to_target(
        self,
        X: pd.DataFrame,
        interpolate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        For panel data, interpolation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            X: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with interpolated columns.
        """
        if not interpolate_keys:
            return X

        result = X.copy()

        # Cas séries temporelles : interpolation globale
        if not is_panel:
            columns = self.extract_column_names(interpolate_keys)
            target_freq = normalize_frequency(target_frequency) if isinstance(target_frequency, str) else target_frequency

            for col in columns:
                if col not in X.columns:
                    continue
                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    X[col], target_freq, method='linear'
                )
                result[col] = interpolated.reindex(X.index)

            return result

        # Cas panel : interpolation par entité
        grouped = self.group_keys_by_entity_and_variable(interpolate_keys)

        for entity, cols in grouped.items():
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            entity_mask = self.get_entity_mask(X, entity)

            for col in cols:
                if col not in X.columns:
                    continue

                entity_series = X.loc[entity_mask, col]
                entity_series = entity_series.droplevel(
                    list(range(X.index.nlevels - 1))
                )

                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    entity_series, entity_target, method='linear'
                )

                reindexed = interpolated.reindex(entity_series.index)
                result.loc[entity_mask, col] = reindexed.values

        return result

    def extract_column_names(
        self,
        keys: List[Union[str, Tuple]],
    ) -> List[str]:
        """Extract unique column names from variable keys.

        Args:
            keys: List of variable identifiers (column names or
                (entity, column) tuples).

        Returns:
            List of unique column names.
        """
        column_names = set()
        for key in keys:
            if isinstance(key, tuple):
                column_names.add(key[-1])
            else:
                column_names.add(key)
        return list(column_names)
