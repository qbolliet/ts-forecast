"""Period position conversion utilities for time series processing.

This module provides the PeriodPositionConverter class to handle conversions
between start and end period positions for time series data.
"""
# Importation des modules
import pandas as pd
from typing import Union, Any
from pandas.tseries.frequencies import to_offset

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import du normalizer et des types
from .normalizer import PeriodPositionNormalizer, PositionType, UserPositionType


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
        # Instance du normaliseur de positions
        self._normalizer = PeriodPositionNormalizer()

    # Méthode principale de conversion
    def convert(
        self,
        value: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex],
        from_unit: Union[PositionType, UserPositionType],
        to_unit: Union[PositionType, UserPositionType],
        freq: str = None,
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
        # Normalisation des positions
        from_code = self._normalizer.normalize(from_unit)
        to_code = self._normalizer.normalize(to_unit)

        # Si les positions sont identiques, retourner les données telles quelles
        if from_code == to_code:
            return value

        # Inférence de la fréquence si non fournie
        if freq is None:
            freq = self._infer_frequency(value)
            if freq is None:
                raise ValueError("Cannot infer frequency from data. Please provide 'freq' parameter.")

        # Conversion selon le type de valeur
        if isinstance(value, pd.DatetimeIndex):
            return self._convert_datetime_index(value, from_code, to_code, freq)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            return self._convert_time_series(value, from_code, to_code, freq)
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
        from_code = self._normalizer.normalize(from_unit)
        to_code = self._normalizer.normalize(to_unit)

        # Retour du facteur (1 si identique, -1 si différent pour indiquer un shift)
        return 1.0 if from_code == to_code else -1.0

    # Méthode auxiliaire d'inférence de la fréquence depuis les données
    def _infer_frequency(self, value: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]) -> str:
        """Infer frequency from time series data or DatetimeIndex.

        Args:
            value: Data from which to infer frequency

        Returns:
            Inferred frequency string, or None if cannot infer

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
            >>> converter._infer_frequency(dates)
            'MS'
        """
        # Extraction de l'index selon le type
        if isinstance(value, pd.DatetimeIndex):
            index = value
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            if isinstance(value.index, pd.DatetimeIndex):
                index = value.index
            else:
                return None
        else:
            return None

        # Inférence de la fréquence pandas
        try:
            inferred = pd.infer_freq(index)
            return inferred
        except Exception:
            # Si l'inférence échoue, essayer d'utiliser la fréquence de l'index
            if hasattr(index, 'freq') and index.freq is not None:
                return index.freq.freqstr if hasattr(index.freq, 'freqstr') else str(index.freq)
            return None

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
        # Décomposition de la fréquence pour obtenir la base fréquence
        base_freq, _ = self._normalizer.decompose_offset(freq)

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
        """
        # Décomposition de l'offset
        freq, from_pos = self._normalizer.decompose_offset(offset_str)

        # Normalisation de la position cible
        to_pos = self._normalizer.normalize(to_position)

        # Si les positions sont identiques, retourner tel quel
        if from_pos == to_pos:
            return offset_str

        # Recombinaison avec la nouvelle position
        return self._normalizer.combine_frequency_position(freq, to_pos)

