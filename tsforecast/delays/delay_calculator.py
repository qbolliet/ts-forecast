"""Median delay calculator for economic indicators.

This module provides tools for analyzing publication delays from DataFrames
and calculating median delays by indicator or entity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any
from datetime import datetime, timedelta
import warnings


class ReleaseDelayCalculator:
    """Median delay calculator from stored data.

    This class analyzes publication delays from DataFrame data
    to calculate delay statistics by indicator and/or entity.

    Args:
        delay_data: DataFrame containing delay records
        default_reference_point: Default reference point ('start' or 'end')
        min_observations: Minimum number of observations to calculate delay
        cache_results: If True, cache results for better performance

    Attributes:
        delay_data_ (pd.DataFrame): Delay records DataFrame
        default_reference_point_ (str): Default reference point
        min_observations_ (int): Minimum number of observations required
        cache_results_ (bool): Cache results activation
        _cache (Dict): Internal results cache

    Examples:
        >>> from tsforecast.delays import ReleaseDelayCalculator
        >>>
        >>> # Create with delay data
        >>> delay_df = pd.DataFrame({...})
        >>> calculator = ReleaseDelayCalculator(delay_data=delay_df)
        >>>
        >>> # Calculate median delays
        >>> delays_dict = calculator.calculate_median_delays()
        >>> print(delays_dict)
        >>> {'GDP': 45.0, 'inflation': 30.0, ...}
        >>>
        >>> # Calculate for panel data
        >>> panel_delays = calculator.calculate_median_delays(
        ...     group_by_entity=True,
        ...     reference_point='end'
        ... )
        >>> print(panel_delays)
        >>> {('France', 'GDP'): 42.0, ('Germany', 'GDP'): 38.0, ...}
    """

    def __init__(self,
                 delay_data: Optional[pd.DataFrame] = None,
                 default_reference_point: str = 'end',
                 min_observations: int = 5,
                 cache_results: bool = True):
        """Initialize delay calculator.

        Args:
            delay_data: DataFrame with delay records
            default_reference_point: 'start' or 'end' for delay calculation
            min_observations: Minimum number of observations required
            cache_results: Activate results cache
        """
        self.delay_data_ = delay_data if delay_data is not None else pd.DataFrame()
        self.default_reference_point_ = default_reference_point
        self.min_observations_ = min_observations
        self.cache_results_ = cache_results
        self._cache = {}

        # Validation du point de référence
        if default_reference_point not in ['start', 'end']:
            raise ValueError("default_reference_point must be 'start' or 'end'")

        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")

    def calculate_median_delays(self,
                               reference_point: Optional[str] = None,
                               group_by_entity: bool = False,
                               indicators: Optional[List[str]] = None,
                               entities: Optional[List[str]] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               data_frequency: Optional[str] = None) -> Dict[Union[str, Tuple[str, str]], float]:
        """Calculate median delays for indicators.

        Args:
            reference_point: Reference point ('start' or 'end'), uses default if None
            group_by_entity: If True, group by (entity, indicator), else by indicator
            indicators: List of indicators to include (all if None)
            entities: List of entities to include (all if None)
            start_date: Start date to filter observations
            end_date: End date to filter observations
            data_frequency: Data frequency to filter

        Returns:
            Dictionary mapping indicator (or (entity, indicator)) to median delay

        Raises:
            ValueError: If no data matches criteria
        """
        reference_point = reference_point or self.default_reference_point_

        # Génération de la clé de cache
        cache_key = self._generate_cache_key(
            reference_point, group_by_entity, indicators, entities,
            start_date, end_date, data_frequency
        )

        # Vérification du cache
        if self.cache_results_ and cache_key in self._cache:
            return self._cache[cache_key]

        # Filtrage des données
        delay_data = self._filter_delay_data(
            reference_point, indicators, entities, start_date, end_date, data_frequency
        )

        if delay_data.empty:
            warnings.warn("No delay data found for specified criteria")
            return {}

        # Calcul des délais médians
        median_delays = self._compute_median_delays(delay_data, group_by_entity)

        # Mise en cache si activée
        if self.cache_results_:
            self._cache[cache_key] = median_delays

        return median_delays

    def calculate_comprehensive_stats(self,
                                    reference_point: Optional[str] = None,
                                    group_by_entity: bool = False,
                                    **kwargs) -> Dict[Union[str, Tuple[str, str]], Dict[str, float]]:
        """Calculate comprehensive delay statistics (median, mean, std, etc.).

        Args:
            reference_point: Reference point for calculation
            group_by_entity: If True, group by (entity, indicator)
            **kwargs: Additional arguments for filtering (same as calculate_median_delays)

        Returns:
            Dictionary mapping indicator to statistics dictionary

        Examples:
            >>> stats = calculator.calculate_comprehensive_stats()
            >>> print(stats['GDP'])
            >>> {
            ...     'median': 45.0,
            ...     'mean': 47.2,
            ...     'std': 12.5,
            ...     'min': 25.0,
            ...     'max': 85.0,
            ...     'count': 120,
            ...     'percentile_25': 38.0,
            ...     'percentile_75': 55.0
            ... }
        """
        reference_point = reference_point or self.default_reference_point_

        # Filtrage des données
        delay_data = self._filter_delay_data(
            reference_point, kwargs.get('indicators'), kwargs.get('entities'),
            kwargs.get('start_date'), kwargs.get('end_date'), kwargs.get('data_frequency')
        )

        if delay_data.empty:
            return {}

        # Groupement selon les critères
        if group_by_entity:
            grouping_cols = ['entity_id', 'indicator_name']
        else:
            grouping_cols = ['indicator_name']

        stats_dict = {}

        for group_values, group_data in delay_data.groupby(grouping_cols):
            if len(group_data) < self.min_observations_:
                continue

            delays = group_data['release_delay_days'].values

            # Calcul des statistiques
            stats = {
                'median': float(np.median(delays)),
                'mean': float(np.mean(delays)),
                'std': float(np.std(delays)),
                'min': float(np.min(delays)),
                'max': float(np.max(delays)),
                'count': len(delays),
                'percentile_25': float(np.percentile(delays, 25)),
                'percentile_75': float(np.percentile(delays, 75))
            }

            # Clé du dictionnaire
            if group_by_entity:
                key = (group_values[0], group_values[1]) if group_values[0] else group_values[1]
            else:
                key = group_values

            stats_dict[key] = stats

        return stats_dict

    def update_delay_data(self, new_delay_records: List[Dict[str, Any]]) -> None:
        """Update delay data with new records.

        Args:
            new_delay_records: List of new delay records to add
        """
        if not new_delay_records:
            return

        new_df = pd.DataFrame(new_delay_records)
        if self.delay_data_.empty:
            self.delay_data_ = new_df
        else:
            self.delay_data_ = pd.concat([self.delay_data_, new_df], ignore_index=True)

        # Nettoyer le cache
        if self.cache_results_:
            self._cache.clear()

    def get_delay_data(self) -> pd.DataFrame:
        """Get the current delay data DataFrame.

        Returns:
            DataFrame with delay data
        """
        return self.delay_data_.copy()

    def _filter_delay_data(self,
                          reference_point: str,
                          indicators: Optional[List[str]] = None,
                          entities: Optional[List[str]] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          data_frequency: Optional[str] = None) -> pd.DataFrame:
        """Filter delay data with specified criteria.

        Args:
            reference_point: Reference point ('start' or 'end')
            indicators: List of indicators to filter
            entities: List of entities to filter
            start_date: Filter start date
            end_date: Filter end date
            data_frequency: Frequency to filter

        Returns:
            DataFrame with filtered delay data
        """
        if self.delay_data_.empty:
            return pd.DataFrame(columns=['indicator_name', 'entity_id', 'observation_date',
                                       'release_delay_days', 'data_frequency', 'is_period_start_reference'])

        filtered_data = self.delay_data_.copy()

        # Filtrage par point de référence
        if 'is_period_start_reference' in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data['is_period_start_reference'] == (reference_point == 'start')
            ]

        # Application des filtres
        if indicators:
            filtered_data = filtered_data[
                filtered_data['indicator_name'].isin(indicators)
            ]

        if entities:
            filtered_data = filtered_data[
                filtered_data['entity_id'].isin(entities)
            ]

        if start_date and 'observation_date' in filtered_data.columns:
            filtered_data = filtered_data[
                pd.to_datetime(filtered_data['observation_date']) >= start_date
            ]

        if end_date and 'observation_date' in filtered_data.columns:
            filtered_data = filtered_data[
                pd.to_datetime(filtered_data['observation_date']) <= end_date
            ]

        if data_frequency and 'data_frequency' in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data['data_frequency'] == data_frequency
            ]

        return filtered_data

    def _compute_median_delays(self,
                              delay_data: pd.DataFrame,
                              group_by_entity: bool) -> Dict[Union[str, Tuple[str, str]], float]:
        """Calcule les délais médians à partir des données récupérées.

        Args:
            delay_data: DataFrame avec les données de délais
            group_by_entity: Si True, groupe par (entity, indicator)

        Returns:
            Dictionnaire des délais médians
        """
        median_delays = {}

        # Définition des colonnes de groupement
        if group_by_entity:
            grouping_cols = ['entity_id', 'indicator_name']
        else:
            grouping_cols = ['indicator_name']

        # Calcul des délais médians par groupe
        for group_values, group_data in delay_data.groupby(grouping_cols):
            if len(group_data) < self.min_observations_:
                continue

            median_delay = group_data['release_delay_days'].median()

            # Définition de la clé
            if group_by_entity:
                if isinstance(group_values, tuple):
                    key = group_values if group_values[0] else group_values[1]
                else:
                    key = group_values
            else:
                key = group_values

            median_delays[key] = float(median_delay)

        return median_delays

    def _generate_cache_key(self, *args) -> str:
        """Génère une clé de cache basée sur les arguments.

        Args:
            *args: Arguments à hasher pour créer la clé

        Returns:
            Clé de cache sous forme de string
        """
        # Conversion des arguments en string pour hashing
        key_components = []
        for arg in args:
            if isinstance(arg, datetime):
                key_components.append(arg.isoformat())
            elif isinstance(arg, list):
                key_components.append('|'.join(str(x) for x in sorted(arg)))
            else:
                key_components.append(str(arg))

        return '_'.join(key_components)


    def clear_cache(self) -> None:
        """Clear results cache."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information.

        Returns:
            Dictionary with cache information
        """
        return {
            'cache_enabled': self.cache_results_,
            'cached_entries': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }

    def set_delay_data(self, delay_data: pd.DataFrame) -> None:
        """Set delay data DataFrame.

        Args:
            delay_data: DataFrame with delay records
        """
        self.delay_data_ = delay_data.copy()
        # Nettoyer le cache
        if self.cache_results_:
            self._cache.clear()