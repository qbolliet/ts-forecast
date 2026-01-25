"""Frequency conversion utilities for time series data.

This module provides the FrequencyConverter class to handle conversions between
different time frequencies using pandas built-in functionality (asfreq and resample).
"""
# Importation des modules
import pandas as pd
from typing import Union, Optional, Literal, Dict, Tuple, List
from pandas.tseries.frequencies import to_offset

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import des utilitaires de fréquence
from .normalizer import FrequencyType, UserFrequencyType
from .utils import normalize_frequency, is_higher_frequency, get_frequency_order
from ..validation import validate_temporal_data

# Import de l'utilitaire de gestion des positions
from ..position.normalizer import PeriodPositionNormalizer

# Types pour les méthodes d'agrégation et d'interpolation
AggregationMethod = Literal['mean', 'sum', 'first', 'last', 'min', 'max', 'median', 'std', 'count']
InterpolationMethod = Literal['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']


# Classe de conversion d'une fréquence dans une autre
class FrequencyConverter(TemporalConverter):
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
        # Initialisation du normalisateur de positions pour gérer les positions S/E
        self._position_normalizer = PeriodPositionNormalizer()

        # Initialisation du convertisseur de durées pour les facteurs de conversion
        from ..duration.converter import DurationConverter
        self._duration_converter = DurationConverter()

    # Implémentation de la méthode abstraite convert de TemporalConverter
    def convert(self,
                value: Union[pd.Series, pd.DataFrame],
                from_unit: str,
                to_unit: str,
                **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Convert data from one frequency to another.

        Implementation of TemporalConverter.convert() for frequencies.

        Args:
            value: Time series data to convert (Series or DataFrame)
            from_unit: Source frequency (not used, frequency is auto-detected)
            to_unit: Target frequency
            **kwargs: Additional conversion parameters (method, fill_method, etc.)

        Returns:
            Converted time series data

        Raises:
            ValueError: If conversion parameters are invalid

        Examples:
            >>> converter = FrequencyConverter()
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
            >>> monthly = converter.convert(series, 'daily', 'monthly', method='mean')
            >>> len(monthly)
            1
        """
        # Redirection vers convert_frequency qui contient toute la logique
        return self.convert_frequency(data=value, target_freq=to_unit, **kwargs)

    # Méthode de conversion d'une fréquence en une autre
    def convert_frequency(self,
                         data: Union[pd.Series, pd.DataFrame],
                         target_freq: Union[str, Dict[str, str]],
                         method: Union[AggregationMethod, InterpolationMethod] = 'mean',
                         fill_method: Optional[str] = None,
                         alignment_method: Literal['ffill', 'bfill', 'nearest', 'none'] = 'ffill',
                         time_col: Optional[str]=None,
                         panel_cols: Optional[List[str]] = None,
                         target_position: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
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
            target_position: Optional position for target frequency ('S', 'E', 'start', 'end').
                If None, preserves source position when identifiable, otherwise uses default 'E'

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
        data = self._validate_conversion_params(data=data, target_freq=target_freq, time_col=time_col, panel_cols=panel_cols)

        # Cas 1: Traitement des Series
        if isinstance(data, pd.Series):
            # Import local pour éviter l'import circulaire
            from ...frequency.detector import detect_frequency

            # Détection de la fréquence actuelle (avec position et anchor)
            detected_freq = detect_frequency(data=data, literal=False)
            if not detected_freq:
                raise ValueError("Cannot detect current frequency of the data")

            # Extraction de la fréquence de base et de la position en utilisant le normalisateur
            # Split pour supprimer l'anchor (-DEC, -OCT, etc.) avant décomposition
            detected_freq_base = detected_freq.split('-')[0]
            current_freq_base, source_position = self._position_normalizer.decompose_offset(detected_freq_base)

            # Normalisation de la fréquence cible (doit être str pour Series)
            if isinstance(target_freq, dict):
                raise ValueError("target_freq must be a string for Series input")

            # Décomposition de la fréquence cible si elle contient déjà une position
            target_freq_base, target_pos_in_arg = self._position_normalizer.decompose_offset(target_freq)

            # Normalisation de la fréquence de base cible
            target_freq_normalized = normalize_frequency(target_freq_base)

            # Détermination de la position cible selon la priorité :
            # 1. target_position explicite (argument)
            # 2. Position dans target_freq si spécifiée (target_pos_in_arg != 'E' ou explicitement dans target_freq)
            # 3. Position source si identifiable (source_position)
            # 4. Convention par défaut 'E'
            if target_position is not None:
                # Cas 1: position explicitement fournie en argument
                final_position = self._position_normalizer.normalize(target_position)
            elif target_freq != target_freq_base:
                # Cas 2: la target_freq contient déjà une position (ex: 'MS', 'QE')
                final_position = target_pos_in_arg
            else:
                # Cas 3: préserver la position source si identifiable, sinon 'E'
                final_position = source_position

            # Construction de la fréquence cible complète avec position
            target_freq_with_position = self._position_normalizer.combine_frequency_position(
                target_freq_normalized,
                final_position
            )

            # Si les fréquences sont identiques (base + position), retourner les données telles quelles
            current_freq_with_position = self._position_normalizer.combine_frequency_position(
                normalize_frequency(current_freq_base),
                source_position
            )
            if current_freq_with_position == target_freq_with_position:
                return data

            # Détermination de la direction de conversion
            if is_higher_frequency(target_freq_normalized, current_freq_base):
                return self._upsample(
                    data=data,
                    target_freq=target_freq_with_position,
                    method=method,
                    fill_method=fill_method
                )
            else:
                return self._downsample(
                    data=data,
                    target_freq=target_freq_with_position,
                    method=method
                )

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
            
    # Implémentation de la méthode abstraite get_conversion_factor de TemporalConverter
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Get approximate conversion factor between two frequencies.

        Implementation of TemporalConverter.get_conversion_factor() for frequencies.

        Note: Frequency conversion factors are approximate and depend on the
        specific time periods involved. This method provides rough estimates.

        Args:
            from_unit: Source frequency
            to_unit: Target frequency

        Returns:
            Approximate conversion factor

        Raises:
            ValueError: If frequencies are not supported

        Examples:
            >>> converter = FrequencyConverter()
            >>> converter.get_conversion_factor('daily', 'monthly')
            30.0
            >>> converter.get_conversion_factor('monthly', 'quarterly')
            3.0
        """
        # Normalisation des fréquences
        from_freq = normalize_frequency(from_unit)
        to_freq = normalize_frequency(to_unit)

        # Obtention des ordres pour calcul approximatif
        from_order = get_frequency_order(from_freq)
        to_order = get_frequency_order(to_freq)

        # Facteurs approximatifs basés sur les ordres
        # (cette méthode est approximative car les fréquences ne se convertissent pas linéairement)
        if from_order == to_order:
            return 1.0
        elif from_order < to_order:
            # Upsampling: from plus granulaire vers moins granulaire
            return float(to_order - from_order)
        else:
            # Downsampling: from moins granulaire vers plus granulaire
            return 1.0 / float(from_order - to_order)

    # Méthode d'agrégation à une fréquence plus faible
    def aggregate_to_lower_frequency(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   target_freq: str,
                                   method: AggregationMethod = 'mean') -> Union[pd.Series, pd.DataFrame]:
        """Aggregate data to a lower frequency using resample.

        Args:
            data: Time series data to aggregate
            target_freq: Target frequency (must be lower than current), with optional position ('MS', 'QE', etc.)
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
        # Validation que target_freq est un offset pandas valide
        # On ne normalise plus la fréquence pour préserver la position (S/E)
        try:
            to_offset(target_freq)
        except Exception as e:
            raise ValueError(f"Invalid target frequency '{target_freq}': {e}")

        # Resampling à la bonne fréquence (avec position préservée)
        resampled = data.resample(target_freq)

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
            target_freq: Target frequency (must be higher than current), with optional position ('MS', 'QE', etc.)
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
        # Validation que target_freq est un offset pandas valide
        # On ne normalise plus la fréquence pour préserver la position (S/E)
        try:
            to_offset(target_freq)
        except Exception as e:
            raise ValueError(f"Invalid target frequency '{target_freq}': {e}")

        # Détection de la fréquence source à partir de l'index
        source_freq = data.index.inferred_freq
        if not source_freq:
            # Tentative de déduction à partir de l'index
            try:
                source_freq = pd.infer_freq(data.index)
            except Exception:
                source_freq = None

        # Si on peut détecter la fréquence source, on étend l'index pour inclure toutes les périodes
        if source_freq:
            # Extension de l'index pour inclure toutes les périodes intermédiaires
            extended_index = self._extend_index_for_upsampling(
                original_index=data.index,
                source_freq=source_freq,
                target_freq=target_freq
            )

            # Réindexation des données sur l'index étendu
            upsampled = data.reindex(extended_index)
        else:
            # Fallback : utilisation de asfreq si la fréquence source n'est pas détectable
            upsampled = data.asfreq(target_freq)

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

    # Méthode d'alignement des fréquences du plusieurs jeux de données
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

        # Import local pour éviter l'import circulaire
        from ...frequency.detector import detect_frequency

        # Détection des fréquences actuelles
        current_freqs = []
        for dataset in datasets:
            freq = detect_frequency(data=dataset, time_col=None,
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
            # Validation de la fréquence en décomposant d'abord pour gérer les positions S/E
            try:
                # Décomposition de la fréquence pour extraire la base et la position
                freq_base, freq_pos = self._position_normalizer.decompose_offset(target_freq)
                # Normalisation de la fréquence de base uniquement
                normalize_frequency(freq_base)
                # Validation de la position si elle est spécifiée et non-default
                if freq_pos and not self._position_normalizer.validate(freq_pos):
                    raise ValueError(f"Invalid position '{freq_pos}' in target frequency '{target_freq}'")
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
                    # Décomposition de la fréquence pour extraire la base et la position
                    freq_base, freq_pos = self._position_normalizer.decompose_offset(freq)
                    # Normalisation de la fréquence de base uniquement
                    normalize_frequency(freq_base)
                    # Validation de la position si elle est spécifiée et non-default
                    if freq_pos and not self._position_normalizer.validate(freq_pos):
                        raise ValueError(f"Invalid position '{freq_pos}' in frequency '{freq}'")
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
            # Décomposition pour gérer les positions S/E
            target_freq_base, target_freq_pos = self._position_normalizer.decompose_offset(target_freq)
            target_freq_normalized = normalize_frequency(target_freq_base)
            # Reconstruction avec position si présente
            if target_freq_pos and target_freq_pos != 'E':
                target_freq_full = self._position_normalizer.combine_frequency_position(target_freq_normalized, target_freq_pos)
            else:
                target_freq_full = target_freq_normalized

            for col in list(data.columns):
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Création de l'association pour la colonne
                if current_freq:
                    frequency_map[col] = (current_freq, target_freq_full)
        else:
            # Fréquences cibles spécifiques par colonne
            for col, target in target_freq.items():
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Création de l'association pour la colonne
                if current_freq:
                    # Décomposition pour gérer les positions S/E
                    target_base, target_pos = self._position_normalizer.decompose_offset(target)
                    target_freq_normalized = normalize_frequency(target_base)
                    # Reconstruction avec position si présente
                    if target_pos and target_pos != 'E':
                        target_freq_full = self._position_normalizer.combine_frequency_position(target_freq_normalized, target_pos)
                    else:
                        target_freq_full = target_freq_normalized

                    frequency_map[col] = (current_freq, target_freq_full)

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
        # Import local pour éviter l'import circulaire
        from ...frequency.detector import detect_frequency

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

            # Décomposition des fréquences pour extraire les bases (sans positions ni anchors)
            source_freq_clean = source_freq.split('-')[0]
            target_freq_clean = target_freq.split('-')[0]
            source_base, _ = self._position_normalizer.decompose_offset(source_freq_clean)
            target_base, _ = self._position_normalizer.decompose_offset(target_freq_clean)

            # Détermination de la direction de conversion (basée sur les fréquences de base)
            if is_higher_frequency(target_base, source_base):
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

    # Méthode auxiliaire d'extension de l'index pour l'upsampling
    def _extend_index_for_upsampling(self,
                                     original_index: pd.DatetimeIndex,
                                     source_freq: str,
                                     target_freq: str) -> pd.DatetimeIndex:
        """Extend index to include all periods when upsampling between frequencies.

        Cette méthode gère l'extension de la plage temporelle lors de l'upsampling
        pour s'assurer que toutes les périodes intermédiaires sont incluses.

        Args:
            original_index: Index original de la série temporelle
            source_freq: Fréquence source avec position (ex: 'QE', 'QS')
            target_freq: Fréquence cible avec position (ex: 'ME', 'MS')

        Returns:
            Index étendu incluant toutes les périodes

        Examples:
            >>> # QE to ME: 4 quarters -> 12 months
            >>> qe_index = pd.date_range('2024-03-31', periods=4, freq='QE')
            >>> extended = _extend_index_for_upsampling(qe_index, 'QE', 'ME')
            >>> len(extended)
            12
        """
        # Extraction des informations de fréquence et position
        # IMPORTANT : source_freq peut contenir un anchor (ex: 'QS-OCT'), il faut le supprimer
        source_freq_clean = source_freq.split('-')[0]
        source_base, source_pos = self._position_normalizer.decompose_offset(source_freq_clean)
        target_base, target_pos = self._position_normalizer.decompose_offset(target_freq)

        # Vérification si extension nécessaire
        # On étend seulement si les bases de fréquence sont différentes et compatibles
        # Ex: Q->M nécessite extension, mais M->M ne nécessite pas extension
        if source_base == target_base:
            # Même fréquence de base, pas d'extension nécessaire
            return original_index

        # Calcul dynamique du ratio de conversion en utilisant DurationConverter
        # Cela garantit la cohérence avec les facteurs de conversion du reste du package
        # Ex: 1 trimestre (Q) = 3 mois (M) → ratio = 3.0
        try:
            # Récupération du facteur de conversion depuis DurationConverter
            ratio = self._duration_converter.get_conversion_factor(source_base, target_base)

            # Vérification que le ratio est un entier positif (ou proche d'un entier)
            # Pour l'extension d'index, on a besoin d'un ratio entier
            if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
                # Le ratio n'est pas un entier ou est < 1, pas d'extension possible
                return original_index

            # Conversion en entier pour l'utilisation dans l'extension
            multiplier = int(round(ratio))
        except (ValueError, KeyError):
            # Si la paire (source_base, target_base) n'est pas supportée par DurationConverter
            # retourner l'index original sans extension
            return original_index

        # Détermination de la plage complète en fonction de la position
        # IMPORTANT: pd.Period() n'accepte pas les suffixes S/E, il faut utiliser les fréquences de base
        # On utilise donc source_base et target_base pour créer les périodes
        start_date = original_index[0]
        end_date = original_index[-1]

        # Conversion en périodes pandas en utilisant la fréquence de BASE (sans S/E)
        # Pour déterminer les bornes de la plage étendue
        try:
            start_period = pd.Period(start_date, freq=source_base)
            end_period = pd.Period(end_date, freq=source_base)
        except Exception:
            # Si la création de Period échoue, retourner l'index original
            return original_index

        # Détermination des bornes de la plage étendue en fonction de la position
        if source_pos == 'E' and target_pos == 'E':
            # Source et cible en position 'end'
            # Ex: QE -> ME : étendre depuis le début de la première période jusqu'à la fin de la dernière
            # Dernier trimestre Q4-2024 se termine le 2024-12-31
            # On veut créer : 2024-10-31, 2024-11-30, 2024-12-31
            extended_start = start_period.to_timestamp(how='start')
            extended_end = end_period.to_timestamp(how='end')

        elif source_pos == 'S' and target_pos == 'S':
            # Source et cible en position 'start'
            # Ex: QS -> MS : étendre depuis le début de la première période jusqu'à la fin de la dernière
            # Premier trimestre Q1-2024 commence le 2024-01-01
            # On veut créer : 2024-01-01, 2024-02-01, 2024-03-01
            extended_start = start_period.to_timestamp(how='start')
            extended_end = end_period.to_timestamp(how='end')

        else:
            # Positions mixtes (source=E, target=S ou inverse)
            # Utiliser une approche générique : étendre sur toute la plage
            extended_start = start_period.to_timestamp(how='start')
            extended_end = end_period.to_timestamp(how='end')

        # Création de l'index complet avec la fréquence cible (incluant position S/E)
        try:
            extended_index = pd.date_range(start=extended_start, end=extended_end, freq=target_freq)
        except Exception:
            # En cas d'erreur, retourner l'index original
            return original_index

        return extended_index

