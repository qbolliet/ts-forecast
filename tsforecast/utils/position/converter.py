"""Period position conversion utilities for time series processing.

This module provides the PeriodPositionConverter class to handle conversions
between start and end period positions for time series data.
"""
# Importation des modules
import re
import pandas as pd
from typing import Union, Optional
from pandas.tseries.frequencies import to_offset

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import du normalizer et des types
from .types import PositionType, UserPositionType
from .utils import normalize_position, decompose_offset, combine_frequency_position
# Import des utilitaires de validation
from ..validation import validate_entities_grouped, validate_sorted_within_groups


# Classe de conversion entre positions de période
class PeriodPositionConverter(TemporalConverter):
    """Handle conversions between period positions (start vs end).

    This class manages conversions between start and end period positions for
    time series data. When converting from start to end (or vice versa), the
    dates are adjusted by one period.

    Examples:
        >>> converter = PeriodPositionConverter()
        >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
        >>> series = pd.Series([1, 2, 3], index=dates)
        >>> end_series = converter.convert(series, 'start', 'end', freq='M')
        >>> # Index dates are shifted to end of month
    """

    # Initialisation
    def __init__(self):
        """Initialize conversion utilities and normalizer."""

    # Méthode principale de conversion
    def convert(
        self,
        value: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex],
        from_unit: Union[PositionType, UserPositionType],
        to_unit: Union[PositionType, UserPositionType],
        freq: Optional[str] = None,
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
        """Convert time series data from one period position to another.

        Implementation of TemporalConverter.convert() for period positions.

        Args:
            value: Time series data or DatetimeIndex to convert
            from_unit: Source position ('S', 'E', 'start', 'end')
            to_unit: Target position ('S', 'E', 'start', 'end')
            freq: Frequency of the time series (e.g., 'M', 'Q', 'Y').
                  If None, will attempt to infer from the data.
            **kwargs: Additional parameters (unused for position conversion)

        Returns:
            Converted time series data with adjusted index

        Raises:
            ValueError: If positions or frequency are invalid

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
            >>> series = pd.Series([1, 2, 3], index=dates)
            >>> end_series = converter.convert(series, 'start', 'end', freq='M')
            >>> # Index: 2023-01-31, 2023-02-28, 2023-03-31
        """
        # Import différé du détecteur (voir note en tête de fichier)
        from tsforecast.frequency.detector import detect_index_frequency

        # Normalisation des positions
        from_code = normalize_position(from_unit)
        to_code = normalize_position(to_unit)

        # Si les positions sont identiques, retourner les données telles quelles
        if from_code == to_code:
            return value

        # Conversion selon le type de valeur
        if isinstance(value, pd.DatetimeIndex):
            # Inférence de la fréquence si non fournie (pour DatetimeIndex simple)
            if freq is None:
                freq = detect_index_frequency(index=value, return_format='full')
                if freq is None:
                    raise ValueError("Cannot infer frequency from data. Please provide 'freq' parameter.")
            return self._convert_datetime_index(value, from_code, to_code, freq)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            # Routage selon le type d'index
            if isinstance(value.index, pd.MultiIndex) and value.index.nlevels >= 2 :
                # Pour les panels, l'inférence se fait par groupe dans _convert_panel()
                return self._convert_panel(value, from_code, to_code, freq)
            elif isinstance(value.index, pd.DatetimeIndex):
                # Inférence de la fréquence si non fournie (pour séries temporelles simples)
                if freq is None:
                    freq = detect_index_frequency(index=value.index, return_format='full')
                    if freq is None:
                        raise ValueError("Cannot infer frequency from data. Please provide 'freq' parameter.")
                return self._convert_time_series(value, from_code, to_code, freq)
            else:
                raise ValueError(f"Data must have DatetimeIndex or MultiIndex with datetime at last level")
        else:
            raise ValueError(f"Unsupported value type for conversion: {type(value)}")

    # Méthode de récupération du facteur de conversion
    def get_conversion_factor(
        self,
        from_unit: Union[PositionType, UserPositionType],
        to_unit: Union[PositionType, UserPositionType]
    ) -> float:
        """Get the conversion factor between two positions.

        Implementation of TemporalConverter.get_conversion_factor() for positions.

        Note: For period positions, there is no multiplicative factor. This method
        returns 1.0 if positions are the same, -1.0 if they are different (indicating
        a shift is needed rather than a multiplication).

        Args:
            from_unit: Source position
            to_unit: Target position

        Returns:
            1.0 if same position, -1.0 if different (shift required)

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> converter.get_conversion_factor('start', 'start')
            1.0
            >>> converter.get_conversion_factor('start', 'end')
            -1.0
        """
        # Normalisation des positions
        from_code = normalize_position(from_unit)
        to_code = normalize_position(to_unit)

        # Retour du facteur (1 si identique, -1 si différent pour indiquer un shift)
        return 1.0 if from_code == to_code else -1.0

    # Méthode auxiliaire de conversion d'un DatetimeIndex
    def _convert_datetime_index(
        self,
        index: pd.DatetimeIndex,
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str
    ) -> pd.DatetimeIndex:
        """Convert DatetimeIndex from one position to another.

        Args:
            index: DatetimeIndex to convert
            from_pos: Source position code
            to_pos: Target position code
            freq: Frequency string

        Returns:
            Converted DatetimeIndex

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
            >>> end_dates = converter._convert_datetime_index(dates, 'S', 'E', 'M')
        """
        # Import différé (voir note en tête de fichier)
        from ..frequency import normalize_frequency

        # Décomposition de la fréquence pour obtenir la base fréquence
        base_freq = normalize_frequency(frequency=freq, return_format='base')

        # Conversion en PeriodIndex puis retour en DatetimeIndex
        if from_pos == 'S' and to_pos == 'E':
            # Conversion de start à end : utiliser to_period puis to_timestamp avec 'end'
            period_index = index.to_period(base_freq)
            return period_index.to_timestamp(how='end')
        elif from_pos == 'E' and to_pos == 'S':
            # Conversion de end à start : utiliser to_period puis to_timestamp avec 'start'
            period_index = index.to_period(base_freq)
            return period_index.to_timestamp(how='start')
        else:
            # Cas identique, retourner tel quel
            return index

    # Méthode auxiliaire de conversion d'une Series ou DataFrame
    def _convert_time_series(
        self,
        data: Union[pd.Series, pd.DataFrame],
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str
    ) -> Union[pd.Series, pd.DataFrame]:
        """Convert Series or DataFrame index from one position to another.

        Args:
            data: Time series data to convert
            from_pos: Source position code
            to_pos: Target position code
            freq: Frequency string

        Returns:
            Time series with converted index

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
            >>> series = pd.Series([1, 2, 3], index=dates)
            >>> end_series = converter._convert_time_series(series, 'S', 'E', 'M')
        """
        # Vérification que l'index est un DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for position conversion")

        # Conversion de l'index
        new_index = self._convert_datetime_index(data.index, from_pos, to_pos, freq)

        # Création d'une copie avec le nouvel index
        if isinstance(data, pd.Series):
            return pd.Series(data.values, index=new_index, name=data.name)
        else:
            return pd.DataFrame(data.values, index=new_index, columns=data.columns)

    # Méthode auxiliaire de conversion d'un panel (Series ou DataFrame avec MultiIndex)
    def _convert_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Convert panel data (MultiIndex) from one position to another.

        For panel data with mixed frequencies, each entity can have its own
        frequency. The conversion is applied separately to each entity group.

        Args:
            data: Panel data with MultiIndex (time at last level)
            from_pos: Source position code
            to_pos: Target position code
            freq: Optional frequency string. If None, frequency is inferred
                  separately for each entity group.

        Returns:
            Panel data with converted time index at last level

        Raises:
            ValueError: If validation fails or structure is invalid

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> entities = ['A', 'A', 'A', 'B', 'B', 'B']
            >>> dates = pd.date_range('2023-01-01', periods=6, freq='MS')
            >>> idx = pd.MultiIndex.from_arrays([entities, dates])
            >>> series = pd.Series(range(6), index=idx)
            >>> end_series = converter._convert_panel(series, 'S', 'E')
        """
        # Import différé du détecteur (voir note en tête de fichier)
        from tsforecast.frequency.detector import detect_dataset_frequency, detect_index_frequency

        # Vérification que l'index est bien un MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Panel data must have a MultiIndex")

        # Vérification que le dernier niveau est datetime
        last_level = data.index.get_level_values(-1)
        if not isinstance(last_level, pd.DatetimeIndex):
            raise ValueError(
                "Last level of MultiIndex must be DatetimeIndex for position conversion. "
                f"Got {type(last_level).__name__} instead."
            )

        # Validation de la structure du panel : entités groupées
        if not validate_entities_grouped(data):
            raise ValueError(
                "Panel entities must be grouped (contiguous blocks). "
                "Each entity's observations must be adjacent in the data. "
                "Please sort your data by entity then by time."
            )

        # Validation du tri des dates au sein de chaque groupe
        if not validate_sorted_within_groups(data):
            raise ValueError(
                "Dates must be sorted within each entity group. "
                "Please ensure time values are monotonically increasing within each entity."
            )

        # Extraction des niveaux d'entités (tous sauf le dernier)
        n_levels = data.index.nlevels
        entity_levels_indices = list(range(n_levels - 1))

        # Groupement par entités pour traitement séparé (support fréquences mixtes)
        # On groupe par tous les niveaux sauf le dernier (le temps)
        if n_levels == 2:
            # Un seul niveau d'entité
            groupby_levels = 0
        else:
            # Plusieurs niveaux d'entité
            groupby_levels = entity_levels_indices

        # Liste pour stocker les segments convertis
        converted_segments = []

        # Conversion de chaque groupe séparément (pour gérer les fréquences mixtes)
        for group_keys, group_data in data.groupby(level=groupby_levels, sort=False):
            # Extraction des dates du groupe
            group_dates = group_data.index.get_level_values(-1)

            # Inférence de la fréquence pour ce groupe si non fournie
            group_freq = freq
            if group_freq is None:
                group_freq = detect_index_frequency(index=group_dates, return_format='full')
                if group_freq is None:
                    # Tentative avec les données du groupe
                    group_freq = detect_dataset_frequency(df=group_data, consistency_mode='highest', strict=False)
                if group_freq is None:
                    raise ValueError(
                        f"Cannot infer frequency for entity {group_keys}. "
                        "Please provide 'freq' parameter or ensure data has regular frequency."
                    )

            # Conversion des dates du groupe
            converted_dates = self._convert_datetime_index(group_dates, from_pos, to_pos, group_freq)

            # Reconstruction du MultiIndex pour ce groupe
            if n_levels == 2:
                # Un seul niveau d'entité : création simple
                new_group_index = pd.MultiIndex.from_arrays(
                    [pd.Index([group_keys] * len(converted_dates)), converted_dates],
                    names=data.index.names
                )
            else:
                # Plusieurs niveaux d'entité : reconstruction de tous les niveaux
                entity_arrays = [
                    group_data.index.get_level_values(i) for i in entity_levels_indices
                ]
                entity_arrays.append(converted_dates)
                new_group_index = pd.MultiIndex.from_arrays(
                    entity_arrays,
                    names=data.index.names
                )

            # Création du segment converti
            if isinstance(data, pd.Series):
                converted_segment = pd.Series(
                    group_data.values,
                    index=new_group_index,
                    name=data.name
                )
            else:
                converted_segment = pd.DataFrame(
                    group_data.values,
                    index=new_group_index,
                    columns=data.columns
                )

            converted_segments.append(converted_segment)

        # Concaténation de tous les segments
        result = pd.concat(converted_segments)

        return result

    # Méthode de conversion d'un offset pandas complet
    def convert_offset(
        self,
        offset_str: str,
        to_position: Union[PositionType, UserPositionType]
    ) -> str:
        """Convert pandas DateOffset to a different position.

        Args:
            offset_str: Source pandas DateOffset (e.g., 'MS', 'QE')
            to_position: Target position

        Returns:
            Converted pandas DateOffset string

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> converter.convert_offset('MS', 'end')
            'ME'
            >>> converter.convert_offset('QE', 'start')
            'QS'
            >>> converter.convert_offset('2MS', 'end')
            '2ME'
            >>> converter.convert_offset('M', 'start')
            'MS'
        """
        # Extraction du multiplicateur éventuel (ex: '2MS' -> '2' et 'MS'),
        # pour le conserver tel quel dans le résultat (decompose_offset() le perd)
        multiplier_match = re.match(r'^(\d+)(.+)$', offset_str)
        multiplier, base_offset = multiplier_match.groups() if multiplier_match else ('', offset_str)

        # Décomposition de l'offset (hors multiplicateur)
        freq, _ = decompose_offset(base_offset)

        # Normalisation de la position cible
        to_pos = normalize_position(to_position)

        # Recombinaison avec la nouvelle position : appliquée même si l'offset
        # d'origine n'en portait pas explicitement une (ex: 'M' -> 'ME')
        return f"{multiplier}{combine_frequency_position(freq, to_pos)}"

