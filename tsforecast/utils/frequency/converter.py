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
from ..parse import parse_frequency

# Import des utilitaires de gestion des positions
from ..position.utils import (
    combine_frequency_position,
    decompose_offset,
    normalize_position,
    validate_position,
)

# Import des utilitaires de panel
from ...panel.utils import is_panel_data

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
            **kwargs: Additional conversion parameters (method, limit, etc.)

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
                         alignment_method: Literal['ffill', 'bfill', 'nearest', 'none'] = 'ffill',
                         time_col: Optional[str]=None,
                         panel_cols: Optional[List[str]] = None,
                         target_position: Optional[str] = None,
                         full_periods_only: bool = False,
                         limit: Union[int, Literal['default'], None] = 'default',
                         limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
                         limit_area: Optional[Literal['inside', 'outside']] = None) -> Union[pd.Series, pd.DataFrame]:
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
            method: Aggregation method for downsampling or interpolation method
                for upsampling
            alignment_method: Method to align indexes when mixing frequencies
                ('ffill', 'bfill', 'nearest', 'none')
            time_col: Identifier of time columns to exclude from conversion
            panel_cols: List of panel identifier columns to exclude from
                conversion
            target_position: Optional position for target frequency ('S', 'E',
                'start', 'end'). If None, preserves source position when
                identifiable, otherwise uses default 'E'
            full_periods_only: If True, periods where the number of non-NaN
                observations is less than expected (based on frequency conversion
                factor) produce NaN during downsampling.
            limit: Maximum number of consecutive NaN values to fill during
                upsampling interpolation. ``'default'`` uses the frequency
                conversion factor. None applies no limit.
            limit_direction: Direction for filling NaN during upsampling. If
                None, defaults to ``'forward'`` for start-positioned frequencies
                and ``'backward'`` for end-positioned frequencies.
            limit_area: Restriction area for filling NaN during upsampling
                (``'inside'`` or ``'outside'``).

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

        # Cas des données de panel (MultiIndex) : traitement entité par entité
        if is_panel_data(data):
            return self._convert_panel_frequency(
                data=data,
                target_freq=target_freq,
                method=method,
                alignment_method=alignment_method,
                target_position=target_position,
                full_periods_only=full_periods_only,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )

        # Cas 1: Traitement des Series
        if isinstance(data, pd.Series):
            # Import local pour éviter l'import circulaire
            from ...frequency.detector import detect_frequency

            # Détection de la fréquence actuelle (avec position et anchor)
            detected_freq = detect_frequency(data=data, return_format='components')

            # Cas où aucune fréquence n'a pu être détectée
            if not detected_freq:
                raise ValueError("Cannot detect current frequency of the data")
            # Parsing des composants
            else :
                source_freq_base, source_position, _ = detected_freq

            # Normalisation de la fréquence cible (doit être str pour Series)
            if isinstance(target_freq, dict):
                raise ValueError("target_freq must be a string for Series input")
            else :
                if target_position is None :
                    target_freq_base, target_position, _ = normalize_frequency(target_freq, return_format='components')
                else :
                    target_freq_base = normalize_frequency(target_freq, return_format='base')

            # Construction de la fréquence cible complète avec position
            target_freq_with_position = combine_frequency_position(
                target_freq_base,
                target_position
            )

            # Si les fréquences sont identiques (base + position), retourner les données telles quelles
            source_freq_with_position = combine_frequency_position(
                source_freq_base,
                source_position
            )
            if source_freq_with_position == target_freq_with_position:
                return data

            # Détermination de la direction de conversion
            if is_higher_frequency(target_freq_base, source_freq_base):
                return self._upsample(
                    data=data,
                    target_freq=target_freq_with_position,
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction,
                    limit_area=limit_area,
                )
            else:
                return self._downsample(
                    data=data,
                    target_freq=target_freq_with_position,
                    method=method,
                    full_periods_only=full_periods_only
                )

        # Cas 2: Traitement des DataFrames
        elif isinstance(data, pd.DataFrame):
            # Construction du frequency_map complet
            frequency_map = self._build_frequency_map(data=data, target_freq=target_freq, target_position=target_position)

            # Groupement des conversions identiques pour optimisation
            grouped_conversions = self._group_conversions_by_operation(frequency_map=frequency_map, method=method)

            # Application des conversions groupées
            result = self._apply_grouped_conversions(
                data=data,
                grouped_conversions=grouped_conversions,
                alignment_method=alignment_method,
                full_periods_only=full_periods_only,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area
            )

            return result

        else:
            raise ValueError("Data must be a pandas Series or DataFrame")
            
    # Implémentation de la méthode abstraite get_conversion_factor de TemporalConverter
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Get approximate conversion factor between two frequencies.

        Implementation of TemporalConverter.get_conversion_factor() for frequencies.
        The factor represents how many ``from_unit`` periods fit into one
        ``to_unit`` period (when ``to_unit`` is lower frequency) or the
        inverse (when ``to_unit`` is higher frequency).

        Args:
            from_unit: Source frequency (e.g., 'daily', 'D', 'monthly', 'M').
            to_unit: Target frequency (e.g., 'monthly', 'M', 'quarterly', 'Q').

        Returns:
            Approximate conversion factor. When converting from a higher
            frequency to a lower one the factor is >= 1 (e.g., daily→monthly
            ≈ 30). When converting from lower to higher the factor is < 1.

        Raises:
            ValueError: If frequencies are not supported

        Examples:
            >>> converter = FrequencyConverter()
            >>> converter.get_conversion_factor('daily', 'monthly')
            30.0
            >>> converter.get_conversion_factor('monthly', 'quarterly')
            3.0
            >>> converter.get_conversion_factor('quarterly', 'monthly')
            0.3333333333333333
        """
        # Normalisation des fréquences de base (sans positions S/E)
        from_freq = normalize_frequency(from_unit)
        to_freq = normalize_frequency(to_unit)

        # Même fréquence → facteur 1
        if from_freq == to_freq:
            return 1.0

        # Délégation au DurationConverter avec arguments inversés :
        # DurationConverter.get_conversion_factor(a, b) = durée(a) / durée(b)
        # = "combien de b dans un a"
        # On veut : "combien de from_freq dans un to_freq" = durée(to) / durée(from)
        # → on passe (to_freq, from_freq)
        return self._duration_converter.get_conversion_factor(to_freq, from_freq)

    # Méthode d'agrégation à une fréquence plus faible
    def aggregate_to_lower_frequency(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   target_freq: str,
                                   method: AggregationMethod = 'mean',
                                   full_periods_only: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """Aggregate data to a lower frequency using resample.

        Args:
            data: Time series data to aggregate
            target_freq: Target frequency (must be lower than current), with
                optional position ('MS', 'QE', etc.)
            method: Aggregation method
            full_periods_only: If True, periods where the number of non-NaN
                observations is strictly less than the expected sub-period count
                (derived from the frequency conversion factor) are set to NaN.
                For example, when aggregating monthly data to quarterly, a
                quarter with only 2 valid months out of 3 will produce NaN.

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
            >>> # Périodes incomplètes remplacées par NaN
            >>> monthly_dates = pd.date_range('2023-01-31', periods=5, freq='ME')
            >>> monthly_series = pd.Series([1, 2, float('nan'), 4, 5], index=monthly_dates)
            >>> quarterly = converter.aggregate_to_lower_frequency(
            ...     monthly_series, 'QE', 'sum', full_periods_only=True
            ... )
        """
        # Importation de la fonction de détection de fréquences
        from tsforecast.frequency.detector import detect_index_frequency

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
            result = resampled.mean()
        elif method == 'sum':
            result = resampled.sum()
        elif method == 'first':
            result = resampled.first()
        elif method == 'last':
            result = resampled.last()
        elif method == 'min':
            result = resampled.min()
        elif method == 'max':
            result = resampled.max()
        elif method == 'median':
            result = resampled.median()
        elif method == 'std':
            result = resampled.std()
        elif method == 'count':
            result = resampled.count()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        # Masquage des périodes incomplètes si demandé
        if full_periods_only:
            # Détection de la fréquence source
            source_freq = detect_index_frequency(index=data.index, return_format='full')

            if source_freq:
                # Extraction des fréquences de base pour le calcul du facteur
                target_base = normalize_frequency(target_freq, return_format='base')
                source_base = normalize_frequency(source_freq, return_format='base')

                try:
                    # Nombre attendu de sous-périodes par période cible
                    expected_count = int(round(
                        self._duration_converter.get_conversion_factor(
                            target_base, source_base
                        )
                    ))
                except (ValueError, KeyError):
                    expected_count = None

                # Application du masque si le facteur a pu être calculé
                if expected_count is not None:
                    valid_counts = resampled.count()
                    result = result.where(valid_counts >= expected_count)

        return result

    # Méthode d'interpolation à une fréquence plus élevée
    def interpolate_to_higher_frequency(self,
                                      data: Union[pd.Series, pd.DataFrame],
                                      target_freq: str,
                                      method: InterpolationMethod = 'linear',
                                      limit: Union[int, str, None] = None,
                                      limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
                                      limit_area: Optional[Literal['inside', 'outside']] = None,
                                      source_freq: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Interpolate data to a higher frequency using asfreq.

        Args:
            data: Time series data to interpolate
            target_freq: Target frequency (must be higher than current), with
                optional position ('MS', 'QE', etc.)
            method: Interpolation method (same semantics as pandas interpolate)
            limit: Maximum number of consecutive NaN values to fill. If
                ``'default'``, uses the frequency conversion factor (e.g. 3 for
                quarterly to monthly). If None, no limit is applied.
            limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            limit_area: Restriction on which NaN values to fill. ``'inside'``
                fills only NaN surrounded by valid values, ``'outside'`` fills
                only NaN outside valid values. See
                :meth:`pandas.DataFrame.interpolate` for details.
            source_freq: Source frequency of the data. If None, it is inferred
                from the index. Provide it explicitly when the index frequency
                differs from the variable's own frequency (e.g. a quarterly
                variable carried on a monthly index).

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
        # Importation de la fonction de détection de fréquence
        from tsforecast.frequency.detector import detect_index_frequency

        # Extraction de la position
        _, target_position, _ = parse_frequency(frequency_str=target_freq)
        # Validation que target_freq est un offset pandas valide
        # On ne normalise plus la fréquence pour préserver la position (S/E)
        try:
            to_offset(target_freq)
        except Exception as e:
            raise ValueError(f"Invalid target frequency '{target_freq}': {e}")

        # Détection de la fréquence source à partir de l'index, sauf si elle est fournie explicitement
        if not source_freq:
           source_freq = detect_index_frequency(index=data.index, return_format='full')

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

        # Résolution de la valeur par défaut de limit
        resolved_limit = self._resolve_interpolation_limit(
            limit=limit,
            source_freq=source_freq,
            target_freq=target_freq
        )

        # Résolution de la valeur par défaut de limit_direction
        resolved_limit_direction = self._resolve_limit_direction(
            limit_direction=limit_direction,
            target_position=target_position,
            target_freq=target_freq
        )

        # Construction des arguments d'interpolation
        interpolate_kwargs = {'method': method}
        if resolved_limit is not None:
            interpolate_kwargs['limit'] = resolved_limit
        if resolved_limit_direction is not None:
            interpolate_kwargs['limit_direction'] = resolved_limit_direction
        if limit_area is not None:
            interpolate_kwargs['limit_area'] = limit_area

        # Application de l'interpolation
        valid_methods = {'linear', 'time', 'index', 'values', 'nearest',
                         'zero', 'slinear', 'quadratic', 'cubic'}
        if method in valid_methods:
            return upsampled.interpolate(**interpolate_kwargs)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}, should be in {valid_methods}")

    # Méthode auxiliaire de résolution de la valeur par défaut de limit
    def _resolve_interpolation_limit(self,
                                     limit: Union[int, str, None],
                                     source_freq: Optional[str],
                                     target_freq: str) -> Optional[int]:
        """Resolve the interpolation limit value.

        When ``limit`` is ``'default'``, computes the frequency conversion
        factor between the source (low) and target (high) frequency.

        Args:
            limit: Raw limit value (int, ``'default'``, or None)
            source_freq: Detected source frequency (may be None)
            target_freq: Target frequency string

        Returns:
            Resolved integer limit or None
        """
        # Retourne le paramètre si ce n'est pas la valeur "default" qui n'est pas tolérée par "interpolate" de pandas
        if limit is None:
            return None
        if isinstance(limit, int):
            return limit

        # Cas 'default' : calcul du facteur de conversion
        if limit == 'default' and source_freq:
            # Normalisation des fréquences
            source_base = normalize_frequency(source_freq, return_format='base')
            target_base = normalize_frequency(target_freq, return_format='base')
            # Calcul du facteur de conversion
            try:
                factor = self._duration_converter.get_conversion_factor(
                    source_base, target_base
                )
                return int(round(factor))
            except (ValueError, KeyError):
                return None

        return None

    # Méthode auxiliaire de résolution de la direction d'interpolation
    def _resolve_limit_direction(self,
                                 limit_direction: Optional[str],
                                 target_position: Optional[str],
                                 target_freq: str) -> Optional[str]:
        """Resolve the default limit_direction based on target position.

        Args:
            limit_direction: Explicit direction or None
            target_position: Target position (``'S'``, ``'E'``, etc.) or None
            target_freq: Target frequency string (used as fallback to extract
                the position)

        Returns:
            Resolved direction string or None
        """
        # Valeur explicite : retour immédiat
        if limit_direction is not None:
            return limit_direction

        # Résolution de la position cible
        position = target_position
        if position is None:
            # Extraction de la position depuis la fréquence cible
            _, extracted_pos = decompose_offset(target_freq)
            position = extracted_pos

        # Normalisation et conversion en direction
        if position is not None:
            normalized_pos = normalize_position(position)
            if normalized_pos == 'S':
                return 'forward'
            elif normalized_pos == 'E':
                return 'backward'

        return None

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
                           return_format='with_position',
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
                freq_base, freq_pos = decompose_offset(target_freq)
                # Normalisation de la fréquence de base uniquement
                normalize_frequency(freq_base)
                # Validation de la position si elle est spécifiée et non-default
                if freq_pos and not validate_position(freq_pos):
                    raise ValueError(f"Invalid position '{freq_pos}' in target frequency '{target_freq}'")
            except ValueError as e:
                raise ValueError(f"Invalid target frequency: {e}")
        elif isinstance(target_freq, dict):
            # Détection de la structure de panel (entités en MultiIndex)
            is_panel = is_panel_data(data)

            # Un dictionnaire n'est autorisé que pour un DataFrame ou un panel (Series MultiIndex)
            if isinstance(data, pd.Series) and not is_panel:
                raise ValueError("Dictionary target_freq is only valid for DataFrame or panel inputs")

            # Colonnes disponibles (hors colonnes de panel)
            if isinstance(data, pd.DataFrame):
                data_cols = set(data.columns)
                if panel_cols:
                    data_cols -= set(panel_cols)
            else:
                data_cols = set()

            # Hors du cas panel : seules des clés de colonnes (str) existantes sont autorisées
            if not is_panel:
                # Rejet des clés tuple (entité / (entité, colonne)) sans structure de panel
                non_column_keys = {k for k in target_freq if not isinstance(k, str)}
                if non_column_keys:
                    raise ValueError(
                        f"Tuple keys in target_freq are only valid for panel inputs: {non_column_keys}"
                    )
                # Vérification que les colonnes désignées existent
                missing_cols = set(target_freq.keys()) - data_cols
                if missing_cols:
                    raise ValueError(f"Columns in target_freq not found in data: {missing_cols}")
            # En panel, les clés peuvent être une colonne (str), une entité (tuple) ou un
            # couple (entité..., colonne) (tuple) : aucune contrainte de sous-ensemble

            # Validation de chaque fréquence cible (clé = colonne, entité ou (entité, colonne))
            for key, freq in target_freq.items():
                try:
                    # Décomposition de la fréquence pour extraire la base et la position
                    freq_base, freq_pos = decompose_offset(freq)
                    # Normalisation de la fréquence de base uniquement
                    normalize_frequency(freq_base)
                    # Validation de la position si elle est spécifiée et non-default
                    if freq_pos and not validate_position(freq_pos):
                        raise ValueError(f"Invalid position '{freq_pos}' in frequency '{freq}'")
                except ValueError as e:
                    raise ValueError(f"Invalid target frequency for key '{key}': {e}")
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

    # Méthode auxiliaire de conversion des données de panel entité par entité
    def _convert_panel_frequency(self,
                                 data: Union[pd.Series, pd.DataFrame],
                                 target_freq: Union[str, Dict[str, str]],
                                 method: str,
                                 alignment_method: str,
                                 target_position: Optional[str],
                                 full_periods_only: bool,
                                 limit: Union[int, Literal['default'], None],
                                 limit_direction: Optional[str],
                                 limit_area: Optional[str]) -> Union[pd.Series, pd.DataFrame]:
        """Convert panel (MultiIndex) data to target frequency, entity by entity.

        Each entity is converted independently on a simple DatetimeIndex, so the
        conversion direction (up/down sampling) is resolved per ``(entity, column)``
        from each entity's own source frequency. This preserves entity-specific
        target frequencies when ``target_freq`` is an ``{entity: freq}`` mapping.

        Args:
            data: Panel data with a MultiIndex (entity levels + time level).
            target_freq: Target frequency (str, {col: freq}, or {entity: freq}).
            method: Aggregation or interpolation method.
            alignment_method: Method to align indexes for mixed frequencies.
            target_position: Optional target position.
            full_periods_only: If True, incomplete periods produce NaN (downsampling).
            limit: Maximum consecutive NaN to fill (upsampling).
            limit_direction: Direction for NaN filling (upsampling).
            limit_area: Restriction area for NaN filling (upsampling).

        Returns:
            Converted panel data with a MultiIndex combining each entity with its
            converted time index.
        """
        # Import local pour éviter les imports circulaires
        from ...panel.utils import get_unique_panel_entities, get_entity_mask

        # Colonnes disponibles (None pour une Series)
        columns = list(data.columns) if isinstance(data, pd.DataFrame) else None
        # Niveaux d'entité (tous sauf le dernier, temporel)
        entity_levels = list(range(data.index.nlevels - 1))
        # Noms des niveaux de l'index d'origine
        index_names = data.index.names

        # Accumulation des résultats par entité
        converted_parts = []

        # Parcours des entités du panel
        for entity in get_unique_panel_entities(data):
            # Masque des observations de l'entité
            entity_mask = get_entity_mask(data, entity)
            # Extraction du sous-jeu de l'entité avec un index temporel simple
            entity_data = data.loc[entity_mask].droplevel(entity_levels)

            # Résolution de la fréquence cible propre à l'entité
            entity_target = self._resolve_panel_target(target_freq, entity, columns)

            # Cas d'une entité sans aucune colonne ciblée : conservation telle quelle
            if isinstance(entity_target, dict) and not entity_target:
                converted_parts.append(data.loc[entity_mask])
                continue

            # Conversion sur l'index simple (chemin non-panel, direction par colonne)
            converted = self.convert_frequency(
                data=entity_data,
                target_freq=entity_target,
                method=method,
                alignment_method=alignment_method,
                target_position=target_position,
                full_periods_only=full_periods_only,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )

            # Reconstruction de l'index MultiIndex en réattachant l'entité
            new_index = pd.MultiIndex.from_tuples(
                [(*entity, time) for time in converted.index],
                names=index_names,
            )
            converted = converted.copy()
            converted.index = new_index

            converted_parts.append(converted)

        # Cas où aucune entité n'a pu être convertie
        if not converted_parts:
            return data

        # Concaténation des entités et tri de l'index
        result = pd.concat(converted_parts)
        result = result.sort_index()

        return result

    # Méthode auxiliaire de résolution de la fréquence cible d'une entité de panel
    def _resolve_panel_target(self,
                              target_freq: Union[str, Dict[str, str]],
                              entity: tuple,
                              columns: Optional[List[str]]) -> Union[str, Dict[str, str]]:
        """Resolve the target frequency applicable to a given panel entity.

        Supports mixed dict keys: ``(entity..., column)`` tuples, ``(entity...)``
        tuples, or plain column names, with precedence
        ``(entity, column)`` > ``(entity,)`` > ``column`` (see
        :func:`resolve_entity_column_frequencies`).

        Args:
            target_freq: Target specification (str or dict with mixed keys).
            entity: Entity tuple.
            columns: Available data columns (None for Series inputs).

        Returns:
            A frequency string (for str inputs or Series entities) or a
            ``{col: freq}`` dict (for DataFrame entities). Columns not covered by
            the mapping are omitted from the dict (left unchanged by the caller).
        """
        # Import local pour éviter les imports circulaires
        from ...panel.utils import (
            get_entity_target_frequency,
            resolve_entity_column_frequencies,
        )

        # Cas d'une chaîne : même cible pour toutes les entités et colonnes
        if not isinstance(target_freq, dict):
            return target_freq

        # Cas Series (pas de colonnes) : résolution vers une fréquence unique par entité
        if columns is None:
            return get_entity_target_frequency(entity, target_freq)

        # Cas DataFrame : résolution par colonne selon la précédence de spécificité
        return resolve_entity_column_frequencies(entity, columns, target_freq)

    # Méthode auxiliaire de construction d'un mapping associant à chaque colonne la
    def _build_frequency_map(self,
                            data: pd.DataFrame,
                            target_freq: Union[str, Dict[str, str]],
                            target_position: Optional[str]) -> Dict[str, Tuple[str, str]]:
        """Build complete frequency map for DataFrame conversion.

        Args:
            data: Input DataFrame
            target_freq: Target frequency (str or dict)

        Returns:
            Dictionary mapping column names to (source_freq, target_freq) tuples
        """
        # Import local pour éviter l'import circulaire
        from ...frequency.detector import detect_dataset_frequency

        # Détection des fréquences source de chaque colonne (index simple → {col: freq})
        # detect_dataset_frequency gère la détection par colonne et la normalisation
        current_frequencies = detect_dataset_frequency(data, return_format='with_position')

        # Construction du frequency_map
        frequency_map = {}

        if isinstance(target_freq, str):
            # Même fréquence cible pour toutes les colonnes
            if target_position is None :
                target_freq_base, resolved_position, _ = normalize_frequency(target_freq, return_format='components')
            else :
                target_freq_base = normalize_frequency(target_freq, return_format='base')
                resolved_position = target_position

            # Construction de la fréquence cible complète avec position
            target_freq_with_position = combine_frequency_position(
                target_freq_base,
                resolved_position
            )

            for col in list(data.columns):
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Création de l'association pour la colonne
                if current_freq:
                    frequency_map[col] = (current_freq, target_freq_with_position)
        else:
            # Fréquences cibles spécifiques par colonne
            for col, target in target_freq.items():
                # Extraction de la fréquence de la colonne
                current_freq = current_frequencies.get(col)
                # Ignore les colonnes absentes du jeu de données ou sans fréquence détectée
                if not current_freq:
                    continue

                # Résolution de la position cible propre à la colonne (variable locale
                # pour ne pas propager la position d'une colonne à la suivante)
                if target_position is None :
                    col_freq_base, col_position, _ = normalize_frequency(target, return_format='components')
                else :
                    col_freq_base = normalize_frequency(target, return_format='base')
                    col_position = target_position

                # Construction de la fréquence cible complète avec position
                col_target_with_position = combine_frequency_position(
                    col_freq_base,
                    col_position
                )
                frequency_map[col] = (current_freq, col_target_with_position)

        return frequency_map

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
                                  alignment_method: str,
                                  full_periods_only: bool = False,
                                  limit: Union[int, str, None] = None,
                                  limit_direction: Optional[str] = None,
                                  limit_area: Optional[str] = None) -> pd.DataFrame:
        """Apply grouped conversions efficiently with proper index alignment.

        Args:
            data: Input DataFrame
            grouped_conversions: Dictionary of grouped conversions
            alignment_method: Method to align indexes when mixing frequencies
            full_periods_only: If True, incomplete periods produce NaN
                (downsampling only)
            limit: Maximum number of consecutive NaN to fill (upsampling only)
            limit_direction: Direction for NaN filling (upsampling only)
            limit_area: Restriction area for NaN filling (upsampling only)

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
            source_base = normalize_frequency(source_freq, return_format='base')
            target_base = normalize_frequency(target_freq, return_format='base')

            # Détermination de la direction de conversion (basée sur les fréquences de base)
            if is_higher_frequency(target_base, source_base):
                # Upsampling
                converted = self._upsample(
                    data=subset, target_freq=target_freq,
                    method=conv_method, limit=limit,
                    limit_direction=limit_direction, limit_area=limit_area
                )
            else:
                # Downsampling
                converted = self._downsample(
                    data=subset, target_freq=target_freq,
                    method=conv_method,
                    full_periods_only=full_periods_only
                )

            # Stockage des colonnes converties
            for col in columns:
                if isinstance(converted, pd.Series):
                    converted_columns[col] = converted
                else:
                    converted_columns[col] = converted[col]

        # Cas où aucune colonne n'a été convertie (ex: fréquence source == cible partout)
        if not converted_columns:
            return data

        # Colonnes présentes mais non converties (à préserver telles quelles)
        non_converted_cols = [col for col in data.columns if col not in columns_to_convert]

        # Vérification si toutes les colonnes converties ont la même fréquence cible
        unique_target_freqs = {target_freq for (_, target_freq, _) in grouped_conversions.keys()}

        # Chemin simple : une seule fréquence cible et aucune colonne à préserver
        if len(unique_target_freqs) == 1 and not non_converted_cols:
            # Reconstruction simple du DataFrame
            result = pd.DataFrame(index=list(converted_columns.values())[0].index)
            for col, conv_series in converted_columns.items():
                result[col] = conv_series
            # Préservation de l'ordre original des colonnes
            return result[[col for col in data.columns if col in result.columns]]

        # Sinon : alignement des fréquences mixtes et/ou conservation des colonnes non converties
        if non_converted_cols:
            base_data = data[non_converted_cols].copy()
        else:
            base_data = pd.DataFrame(index=data.index)

        # Alignement des colonnes converties (et réattachement des colonnes non converties)
        result = self._align_mixed_frequency_columns(
            base_data=base_data,
            converted_columns=converted_columns,
            alignment_method=alignment_method
        )

        # Préservation de l'ordre original des colonnes
        return result[[col for col in data.columns if col in result.columns]]

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
                limit: Union[int, str, None] = None,
                limit_direction: Optional[str] = None,
                limit_area: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Perform upsampling using asfreq and interpolation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Interpolation method
            limit: Maximum number of consecutive NaN to fill (int,
                ``'default'``, or None)
            limit_direction: Direction for NaN filling
            limit_area: Restriction area for NaN filling

        Returns:
            Upsampled data
        """
        # Délégation directe pour les données sans panel
        if not is_panel_data(data):
            return self.interpolate_to_higher_frequency(
                data, target_freq, method,
                limit=limit, limit_direction=limit_direction,
                limit_area=limit_area
            )

        # Données de panel : application par entité via groupby
        panel_levels = list(range(data.index.nlevels - 1))
        return data.groupby(level=panel_levels, group_keys=False).apply(
            lambda x: self.interpolate_to_higher_frequency(
                x.droplevel(panel_levels),
                target_freq, method,
                limit=limit, limit_direction=limit_direction,
                limit_area=limit_area
            )
        )

    # Méthode auxiliaire de diminution de la fréquence par agrégation
    def _downsample(self,
                data: Union[pd.Series, pd.DataFrame],
                target_freq: Union[FrequencyType, UserFrequencyType],
                method: str,
                full_periods_only: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """Perform downsampling using resample and aggregation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Aggregation method
            full_periods_only: If True, incomplete periods produce NaN

        Returns:
            Downsampled data
        """
        # Délégation directe pour les données sans panel
        if not is_panel_data(data):
            return self.aggregate_to_lower_frequency(
                data, target_freq, method, full_periods_only
            )

        # Données de panel : application par entité via groupby
        panel_levels = list(range(data.index.nlevels - 1))
        return data.groupby(level=panel_levels, group_keys=False).apply(
            lambda x: self.aggregate_to_lower_frequency(
                x.droplevel(panel_levels),
                target_freq, method, full_periods_only
            )
        )

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
        source_base, source_pos, _ = normalize_frequency(
            source_freq,
            return_format='components'
        )
        target_base, target_pos, _ = normalize_frequency(target_freq, return_format='components')

        # Définir des valeurs par défaut pour les positions si None (convention: 'E')
        source_pos = source_pos if source_pos is not None else 'E'
        target_pos = target_pos if target_pos is not None else 'E'

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

        except (ValueError, KeyError):
            # Si la paire (source_base, target_base) n'est pas supportée par DurationConverter
            # retourner l'index original sans extension
            return original_index

        # Détermination de la plage complète en fonction de la position
        # IMPORTANT: pd.Period() n'accepte pas les suffixes S/E, il faut utiliser les fréquences de base
        # On utilise donc source_base et target_base pour créer les périodes
        start_date = original_index[0]
        end_date = original_index[-1]

        # Construction du nouvel index
        try:
            # Conversion en périodes pandas en utilisant la fréquence de BASE (sans S/E)
            # Pour déterminer les bornes de la plage étendue
            extended_start = pd.Period(start_date, freq=source_base).to_timestamp(how='start')
            extended_end = pd.Period(end_date, freq=source_base).to_timestamp(how='end')

            # Création de l'index complet avec la fréquence cible (incluant position S/E)
            extended_index = pd.date_range(start=extended_start, end=extended_end, freq=target_freq)
        except Exception:
            # Si la création de Period échoue, retourner l'index original
            return original_index

        return extended_index