"""Frequency conversion utilities for time series data.

This module provides the FrequencyConverter class to handle conversions between
different time frequencies using pandas built-in functionality (asfreq and resample).
"""
# Importation des modules
import pandas as pd
import numpy as np
from typing import Union, Optional, Literal, Dict, Any, Tuple, List
from pandas.tseries.frequencies import to_offset

# Import des utilitaires de fréquence
from .utils import normalize_frequency, is_higher_frequency, get_frequency_order, FrequencyType, UserFrequencyType
from .detector import detect_frequency
from ..utils import validate_temporal_data

# Types pour les méthodes d'agrégation et d'interpolation
AggregationMethod = Literal['mean', 'sum', 'first', 'last', 'min', 'max', 'median', 'std', 'count']
InterpolationMethod = Literal['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']


# Classe de conversion d'une fréquence dans une autre
class FrequencyConverter:
    """Handle conversions between different time frequencies.

    This class manages frequency conversions using pandas built-in functionality,
    primarily asfreq for upsampling and resample for downsampling.

    Examples:
        >>> converter = FrequencyConverter()
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> monthly = converter.convert_frequency(series, 'monthly', method='mean')
        >>> len(monthly)
        1
    """
    # Initialisation
    def __init__(self):
        """Initialize the FrequencyConverter."""
        pass
    
    # Méthode de conversion d'une fréquence en une autre
    def convert_frequency(self,
                         data: Union[pd.Series, pd.DataFrame],
                         target_freq: Union[str, Dict[str, str]],
                         method: Union[AggregationMethod, InterpolationMethod] = 'mean',
                         fill_method: Optional[str] = None,
                         alignment_method: Literal['ffill', 'bfill', 'nearest', 'none'] = 'ffill',
                         time_col: Optional[str]=None,
                         panel_cols: Optional[List[str]] = None) -> Union[pd.Series, pd.DataFrame]:
        """Convert data to target frequency using pandas built-in methods.

        This is the main conversion method that automatically determines whether
        to use upsampling (asfreq) or downsampling (resample) based on the
        frequency relationship. Supports both Series and DataFrame with flexible
        target frequency specification.

        Args:
            data: Time series data to convert (Series or DataFrame)
            target_freq: Target frequency specification:
                - str: Apply same frequency to all columns
                - Dict[str, str]: Map each column to its target frequency
            method: Aggregation method for downsampling or interpolation method for upsampling
            fill_method: Fill method for missing values ('ffill', 'bfill', None)
            alignment_method: Method to align indexes when mixing frequencies ('ffill', 'bfill', 'nearest', 'none')
            time_col: Identifier of time columns to exclude from conversion
            panel_cols: List of panel identifier columns to exclude from conversion

        Returns:
            Converted time series data

        Raises:
            ValueError: If conversion parameters are invalid

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> # Series conversion
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.convert_frequency(daily_series, 'monthly', method='mean')
            >>> isinstance(monthly, pd.Series)
            True
            >>> # DataFrame with string target_freq
            >>> daily_df = pd.DataFrame({'a': range(31), 'b': range(31, 62)}, index=daily_dates)
            >>> monthly_df = converter.convert_frequency(daily_df, 'monthly', method='mean')
            >>> # DataFrame with dict target_freq
            >>> mixed_freq = converter.convert_frequency(daily_df, {'a': 'monthly', 'b': 'weekly'}, method='mean')
        """
        # Validation des paramètres d'entrée
        data = self._validate_conversion_params(data=data, target_freq=target_freq, method=method, panel_cols=panel_cols)

        # Cas 1: Traitement des Series
        if isinstance(data, pd.Series):
            # Détection de la fréquence actuelle
            current_freq = detect_frequency(data=data, literal=False)
            if not current_freq:
                raise ValueError("Cannot detect current frequency of the data")

            # Normalisation de la fréquence cible (doit être str pour Series)
            if isinstance(target_freq, dict):
                raise ValueError("target_freq must be a string for Series input")
            target_freq_normalized = normalize_frequency(target_freq)

            # Si les fréquences sont identiques, retourner les données telles quelles
            if normalize_frequency(current_freq) == target_freq_normalized:
                return data

            # Détermination de la direction de conversion
            if is_higher_frequency(target_freq, current_freq):
                return self._upsample(data=data, target_freq=target_freq_normalized, method=method, fill_method=fill_method)
            else:
                return self._downsample(data=data, target_freq=target_freq_normalized, method=method)

        # Cas 2: Traitement des DataFrames
        elif isinstance(data, pd.DataFrame):
            # Construction du frequency_map complet
            frequency_map = self._build_frequency_map(data=data, target_freq=target_freq)

            # Groupement des conversions identiques pour optimisation
            grouped_conversions = self._group_conversions_by_operation(frequency_map=frequency_map, method=method)

            # Application des conversions groupées
            result = self._apply_grouped_conversions(data=data, grouped_conversions=grouped_conversions, method=method, fill_method=fill_method, alignment_method=alignment_method)

            return result

        else:
            raise ValueError("Data must be a pandas Series or DataFrame")

    # Méthode d'agrégation à une fréquence plus faible
    def aggregate_to_lower_frequency(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   target_freq: str,
                                   method: AggregationMethod = 'mean') -> Union[pd.Series, pd.DataFrame]:
        """Aggregate data to a lower frequency using resample.

        Args:
            data: Time series data to aggregate
            target_freq: Target frequency (must be lower than current)
            method: Aggregation method

        Returns:
            Aggregated time series data

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.aggregate_to_lower_frequency(daily_series, 'monthly', 'sum')
            >>> len(monthly)
            1
        """
        # Normalisation de la fréquence
        target_base_freq = normalize_frequency(target_freq)
        # Resampling à la bonne fréquence
        resampled = data.resample(target_base_freq)

        # Application de la méthode d'agrégation
        if method == 'mean':
            return resampled.mean()
        elif method == 'sum':
            return resampled.sum()
        elif method == 'first':
            return resampled.first()
        elif method == 'last':
            return resampled.last()
        elif method == 'min':
            return resampled.min()
        elif method == 'max':
            return resampled.max()
        elif method == 'median':
            return resampled.median()
        elif method == 'std':
            return resampled.std()
        elif method == 'count':
            return resampled.count()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    # Méthode d'interpolation à une fréquence plus élevée
    def interpolate_to_higher_frequency(self,
                                      data: Union[pd.Series, pd.DataFrame],
                                      target_freq: str,
                                      method: InterpolationMethod = 'linear',
                                      fill_method: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Interpolate data to a higher frequency using asfreq.

        Args:
            data: Time series data to interpolate
            target_freq: Target frequency (must be higher than current)
            method: Interpolation method
            fill_method: Fill method for missing values

        Returns:
            Interpolated time series data

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> monthly_dates = pd.date_range('2023-01-01', periods=3, freq='M')
            >>> monthly_series = pd.Series([10, 20, 30], index=monthly_dates)
            >>> daily = converter.interpolate_to_higher_frequency(monthly_series, 'daily', 'linear')
            >>> len(daily) > len(monthly_series)
            True
        """
        # Normalisation de la fréquence
        target_base_freq = normalize_frequency(target_freq)

        # Utilisation de 'asfreq' pour créer la nouvelle fréquence
        upsampled = data.asfreq(target_base_freq)

        # Application du remplissage si spécifié
        if fill_method == 'ffill':
            upsampled = upsampled.fillna(method='ffill')
        elif fill_method == 'bfill':
            upsampled = upsampled.fillna(method='bfill')

        # Application de l'interpolation
        if method == 'linear':
            return upsampled.interpolate(method='linear')
        elif method == 'time':
            return upsampled.interpolate(method='time')
        elif method == 'index':
            return upsampled.interpolate(method='index')
        elif method == 'values':
            return upsampled.interpolate(method='values')
        elif method == 'nearest':
            return upsampled.interpolate(method='nearest')
        elif method == 'zero':
            return upsampled.interpolate(method='zero')
        elif method == 'slinear':
            return upsampled.interpolate(method='slinear')
        elif method == 'quadratic':
            return upsampled.interpolate(method='quadratic')
        elif method == 'cubic':
            return upsampled.interpolate(method='cubic')
        else:
            return upsampled

    def align_frequencies(self,
                        *datasets: Union[pd.Series, pd.DataFrame],
                        target_freq: Optional[str] = None,
                        method: str = 'mean') -> tuple:
        """Align multiple datasets to the same frequency.

        Args:
            *datasets: Variable number of time series datasets
            target_freq: Target frequency (if None, uses the highest common frequency)
            method: Conversion method to use

        Returns:
            Tuple of aligned datasets

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> monthly_dates = pd.date_range('2023-01-01', periods=2, freq='M')
            >>> daily_data = pd.Series(range(5), index=daily_dates)
            >>> monthly_data = pd.Series([10, 20], index=monthly_dates)
            >>> aligned = converter.align_frequencies(daily_data, monthly_data, target_freq='monthly')
            >>> len(aligned) == 2
            True
        """
        if not datasets:
            return tuple()

        # Détection des fréquences actuelles
        current_freqs = []
        for dataset in datasets:
            freq = detect_frequency(df=dataset, time_col=None,
                           panel_cols= None,
                           literal=False,
                           check_consistency=True,
                           strict=False)
            if freq:
                current_freqs.append(freq)

        if not current_freqs:
            raise ValueError("Cannot detect frequency for any dataset")

        # Détermination de la fréquence cible si non spécifiée
        if target_freq is None:
            # Utilisation de la fréquence la plus basse (moins granulaire)
            freq_orders = {}
            for freq in set(current_freqs):
                freq_orders[freq] = get_frequency_order(freq)

            target_freq = max(freq_orders.keys(), key=lambda x: freq_orders[x])

        # Conversion de tous les datasets vers la fréquence cible
        aligned_datasets = []
        for dataset in datasets:
            aligned = self.convert_frequency(dataset, target_freq, method=method)
            aligned_datasets.append(aligned)

        return tuple(aligned_datasets)

    # Méthode auxiliaire de validation des paramètres
    def _validate_conversion_params(self,
                                  data: Union[pd.Series, pd.DataFrame],
                                  target_freq: Union[str, Dict[str, str]],
                                  time_col: Optional[str]=None,
                                  panel_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate conversion parameters.

        Args:
            data: Input data
            target_freq: Target frequency (str or dict)
            time_col: Time identifier column
            panel_cols: Panel identifier columns

        Returns:
            Validated data
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Vérification du jeu de données
        data = validate_temporal_data(data=data, time_col=time_col, panel_cols=panel_cols, strict=True, sort_data=True, return_metadata=False)

        # Vérification que la fréquence cible est spécifiée
        if not target_freq:
            raise ValueError("Target frequency cannot be empty")

        # Validation de target_freq selon son type
        if isinstance(target_freq, str):
            # Validation simple de la fréquence
            try:
                normalize_frequency(target_freq)
            except ValueError as e:
                raise ValueError(f"Invalid target frequency: {e}")
        elif isinstance(target_freq, dict):
            # Validation pour dictionnaire
            if isinstance(data, pd.Series):
                raise ValueError("Dictionary target_freq is only valid for DataFrame inputs")

            # Vérification que toutes les clés existent dans les colonnes
            data_cols = set(data.columns)
            if panel_cols:
                data_cols -= set(panel_cols)

            target_cols = set(target_freq.keys())
            missing_cols = target_cols - data_cols
            if missing_cols:
                raise ValueError(f"Columns in target_freq not found in data: {missing_cols}")

            # Validation de chaque fréquence cible
            for col, freq in target_freq.items():
                try:
                    normalize_frequency(freq)
                except ValueError as e:
                    raise ValueError(f"Invalid target frequency for column '{col}': {e}")
        else:
            raise ValueError("target_freq must be a string or dictionary")

        # Validation des panel_cols si spécifiés
        if panel_cols:
            # Vérification que les panel_cols ne sont pas dans target_freq si dict
            if isinstance(target_freq, dict):
                overlap = set(panel_cols) & set(target_freq.keys())
                if overlap:
                    raise ValueError(f"Panel columns cannot be in target_freq: {overlap}")
        
        return data

    # Méthode auxiliaire de construction d'un mapping associant à chaque colonne la 
    def _build_frequency_map(self,
                            data: pd.DataFrame,
                            target_freq: Union[str, Dict[str, str]]) -> Dict[str, Tuple[str, str]]:
        """Build complete frequency map for DataFrame conversion.

        Args:
            data: Input DataFrame
            target_freq: Target frequency (str or dict)

        Returns:
            Dictionary mapping column names to (source_freq, target_freq) tuples
        """
        # Détection des fréquences actuelles pour chaque colonne
        current_frequencies = self._detect_column_frequencies(data=data, columns=list(data.columns))

        # Construction du frequency_map
        frequency_map = {}

        if isinstance(target_freq, str):
            # Même fréquence cible pour toutes les colonnes
            target_freq_normalized = normalize_frequency(target_freq)
            for col in list(data.columns):
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Création de l'association pour la colonne
                if current_freq:
                    frequency_map[col] = (current_freq, target_freq_normalized)
        else:
            # Fréquences cibles spécifiques par colonne
            for col, target in target_freq.items():
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Création de l'association pour la colonne
                if current_freq:
                    target_freq_normalized = normalize_frequency(target)
                    frequency_map[col] = (current_freq, target_freq_normalized)

        return frequency_map

    # Méthode auxiliaire de détection des fréquences de colonnes
    def _detect_column_frequencies(self,
                                  data: pd.DataFrame,
                                  columns: List[str]) -> Dict[str, str]:
        """Detect frequencies for specified columns in DataFrame.

        Args:
            data: Input DataFrame
            columns: List of columns to detect frequencies for

        Returns:
            Dictionary mapping column names to detected frequencies
        """
        # Initialisation du dictionnaire résultat
        frequencies = {}
        # Parcours des colonnes
        for col in columns:
            try:
                # Détection de la fréquence
                freq = detect_frequency(data=data[col], literal=False)
                # Association à la colonne
                if freq:
                    frequencies[col] = freq
            except Exception:
                # Si la détection échoue pour une colonne, continuer
                continue

        return frequencies

    # Méthode auxiliaire de groupement des conversions par opération
    def _group_conversions_by_operation(self,
                                       frequency_map: Dict[str, Tuple[str, str]],
                                       method: str) -> Dict[Tuple[str, str, str], List[str]]:
        """Group conversions by identical operations for efficiency.

        Args:
            frequency_map: Dictionary mapping columns to (source_freq, target_freq)
            method: Conversion method

        Returns:
            Dictionary mapping (source_freq, target_freq, method) to list of columns
        """
        # Initialisation du dictionnaire associant une transformation à un ensemble de colonnes
        grouped = {}

        # Parcours du mapping
        for col, (source_freq, target_freq) in frequency_map.items():
            # Ignore les colonnes dont la fréquence ne change pas
            if source_freq == target_freq:
                continue

            # Clé de groupement
            key = (source_freq, target_freq, method)

            # Ajout de la colonne au groupe
            if key not in grouped:
                grouped[key] = []
            # Ajout de la colonne à la clé
            grouped[key].append(col)

        return grouped

    # Méthode auxiliaire d'application des conversions groupées
    def _apply_grouped_conversions(self,
                                  data: pd.DataFrame,
                                  grouped_conversions: Dict[Tuple[str, str, str], List[str]],
                                  method: str,
                                  fill_method: Optional[str],
                                  alignment_method: str) -> pd.DataFrame:
        """Apply grouped conversions efficiently with proper index alignment.

        Args:
            data: Input DataFrame
            grouped_conversions: Dictionary of grouped conversions
            method: Conversion method
            fill_method: Fill method for missing values
            alignment_method: Method to align indexes when mixing frequencies

        Returns:
            DataFrame with all conversions applied and properly aligned
        """
        # Dictionnaire pour stocker les colonnes converties
        converted_columns = {}

        # Ensemble des colonnes qui seront converties
        columns_to_convert = set()
        for columns_list in grouped_conversions.values():
            columns_to_convert.update(columns_list)

        # Traitement de chaque groupe de conversions
        for (source_freq, target_freq, conv_method), columns in grouped_conversions.items():
            # Extraction des colonnes à convertir
            subset = data[columns]

            # Détermination de la direction de conversion
            if is_higher_frequency(target_freq, source_freq):
                # Upsampling
                converted = self._upsample(data=subset, target_freq=target_freq, method=conv_method, fill_method=fill_method)
            else:
                # Downsampling
                converted = self._downsample(data=subset, target_freq=target_freq, method=conv_method)

            # Stockage des colonnes converties
            for col in columns:
                if isinstance(converted, pd.Series):
                    converted_columns[col] = converted
                else:
                    converted_columns[col] = converted[col]

        # Vérification si toutes les colonnes ont la même fréquence cible
        unique_target_freqs = set()
        for (source_freq, target_freq, conv_method) in grouped_conversions.keys():
            unique_target_freqs.add(target_freq)

        # Si une seule fréquence cible, pas besoin d'alignement complexe
        if len(unique_target_freqs) == 1:
            # Reconstruction simple du DataFrame
            result = pd.DataFrame(index=list(converted_columns.values())[0].index)
            for col, conv_series in converted_columns.items():
                result[col] = conv_series
            return result

        # Sinon, utilisation de la méthode d'alignement pour les fréquences mixtes
        else:
            # Création d'un DataFrame de base avec les colonnes non converties
            non_converted_cols = [col for col in data.columns if col not in columns_to_convert]
            if non_converted_cols:
                base_data = data[non_converted_cols].copy()
            else:
                base_data = pd.DataFrame(index=data.index)

            # Alignement des colonnes converties
            result = self._align_mixed_frequency_columns(
                base_data=base_data,
                converted_columns=converted_columns,
                alignment_method=alignment_method
            )

            return result

    # Méthode auxiliaire de détection du MultiIndex pour les données de panel
    def _is_panel_data(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Check if data has MultiIndex structure for panel data.

        Args:
            data: Input data

        Returns:
            True if data has MultiIndex with 2+ levels
        """
        return isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2

    # Méthode auxiliaire d'application du resampling pour les données de panel
    def _apply_panel_resample(self,
                             data: Union[pd.Series, pd.DataFrame],
                             target_freq: str,
                             method: str,
                             operation: Literal['upsample', 'downsample'],
                             fill_method: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Apply resampling to panel data by grouping on panel levels.

        Args:
            data: Panel data with MultiIndex (panel_levels + time_level)
            target_freq: Target frequency
            method: Conversion method
            operation: Type of operation ('upsample' or 'downsample')
            fill_method: Fill method for missing values (upsampling only)

        Returns:
            Resampled panel data
        """
        # Identification des niveaux de panel (tous sauf le dernier qui est le temps)
        panel_levels = list(range(data.index.nlevels - 1))

        # Groupement par entités de panel
        grouped = data.groupby(level=panel_levels, group_keys=False)

        # Application du resampling à chaque groupe
        if operation == 'downsample':
            # Application de l'agrégation
            if isinstance(data, pd.Series):
                resampled = grouped.apply(lambda x: x.droplevel(panel_levels).resample(target_freq).agg(method))
            else:
                resampled = grouped.apply(lambda x: x.droplevel(panel_levels).resample(target_freq).agg(method))
        else:  # upsample
            def upsample_group(x):
                # Suppression des niveaux de panel pour le resampling
                x_dropped = x.droplevel(panel_levels)
                # Application de asfreq
                upsampled = x_dropped.asfreq(target_freq)
                # Application du fill_method si spécifié
                if fill_method == 'ffill':
                    upsampled = upsampled.fillna(method='ffill')
                elif fill_method == 'bfill':
                    upsampled = upsampled.fillna(method='bfill')
                # Interpolation
                if method in ['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
                    upsampled = upsampled.interpolate(method=method)
                return upsampled

            resampled = grouped.apply(upsample_group)

        return resampled

    # Méthode auxiliaire d'alignement des indexes de différentes fréquences
    def _align_mixed_frequency_columns(self,
                                      base_data: pd.DataFrame,
                                      converted_columns: Dict[str, pd.DataFrame],
                                      alignment_method: str) -> pd.DataFrame:
        """Align columns with different frequencies using specified method.

        Args:
            base_data: Original DataFrame with base index
            converted_columns: Dictionary mapping column names to converted Series/DataFrames
            alignment_method: Method to use for alignment ('ffill', 'bfill', 'nearest', 'none')

        Returns:
            DataFrame with aligned columns
        """
        if not converted_columns:
            return base_data

        # Collecte de tous les indexes uniques
        all_indexes = [base_data.index]
        for conv_data in converted_columns.values():
            if isinstance(conv_data, pd.Series):
                all_indexes.append(conv_data.index)
            else:
                all_indexes.append(conv_data.index)

        # Création d'un index unifié (union de tous les indexes)
        unified_index = all_indexes[0]
        for idx in all_indexes[1:]:
            unified_index = unified_index.union(idx)

        # Tri de l'index unifié
        if isinstance(unified_index, pd.MultiIndex):
            unified_index = unified_index.sort_values()
        else:
            unified_index = unified_index.sort_values()

        # Réindexation de toutes les colonnes sur l'index unifié
        result = pd.DataFrame(index=unified_index)

        # Copie des colonnes non converties
        for col in base_data.columns:
            if col not in converted_columns:
                result[col] = base_data[col].reindex(unified_index)

        # Ajout des colonnes converties avec alignement
        for col, conv_data in converted_columns.items():
            if isinstance(conv_data, pd.Series):
                result[col] = conv_data.reindex(unified_index)
            else:
                result[col] = conv_data[col].reindex(unified_index)

            # Application de la méthode d'alignement
            if alignment_method == 'ffill':
                result[col] = result[col].fillna(method='ffill')
            elif alignment_method == 'bfill':
                result[col] = result[col].fillna(method='bfill')
            elif alignment_method == 'nearest':
                result[col] = result[col].interpolate(method='nearest')
            # 'none' ne fait rien, garde les NaN

        return result

    # Méthode auxiliaire d'augmentation de la fréquence par interpolation
    def _upsample(self,
                 data: Union[pd.Series, pd.DataFrame],
                 target_freq: Union[FrequencyType, UserFrequencyType],
                 method: str,
                 fill_method: Optional[str]) -> Union[pd.Series, pd.DataFrame]:
        """Perform upsampling using asfreq and interpolation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Interpolation method
            fill_method: Fill method for missing values

        Returns:
            Upsampled data
        """
        # Vérification si les données sont de type panel (MultiIndex)
        if self._is_panel_data(data):
            return self._apply_panel_resample(data, target_freq, method, 'upsample', fill_method)
        else:
            return self.interpolate_to_higher_frequency(data, target_freq, method, fill_method)

    # Méthode auxiliaire de diminution de la fréquence par agrégation
    def _downsample(self,
                   data: Union[pd.Series, pd.DataFrame],
                   target_freq: Union[FrequencyType, UserFrequencyType],
                   method: str) -> Union[pd.Series, pd.DataFrame]:
        """Perform downsampling using resample and aggregation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Aggregation method

        Returns:
            Downsampled data
        """
        # Vérification si les données sont de type panel (MultiIndex)
        if self._is_panel_data(data):
            return self._apply_panel_resample(data, target_freq, method, 'downsample')
        else:
            return self.aggregate_to_lower_frequency(data, target_freq, method)

