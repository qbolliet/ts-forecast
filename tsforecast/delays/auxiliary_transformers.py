# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Literal, Tuple
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Importation des modules du package
from tsforecast.utils.frequency import to_pandas_freq, normalize_frequency, is_higher_frequency
from tsforecast.utils.frequency.parser import (
    detect_and_parse_frequency,
    build_frequency_string
)
from tsforecast.utils.duration.converter import DurationConverter
from tsforecast.utils.time.utils import get_period_boundaries
from tsforecast.utils.validation import validate_temporal_data


# Transformer 'shiftant' les séries sur un nombre donnée de périodes
class ShiftTransformer(BaseEstimator, TransformerMixin):
    """Shift time series data by N periods with no data loss.

    This transformer extends the index to avoid losing data at boundaries.
    For positive shifts, extends at the beginning; for negative shifts, extends at the end.
    Supports both Series and DataFrame inputs.

    Parameters:
        n_periods: Number of periods to shift (can be negative)
        frequency: Frequency for period arithmetic ('D', 'M', 'Q', 'W', 'h', etc.)
        frequency_check: Frequency validation mode
            - "ignore": No validation (default)
            - "warn": Issue warning if detected frequency differs
            - "raise": Raise error if detected frequency differs

    Attributes:
        n_periods: Stored number of periods to shift
        frequency: Stored frequency code
        frequency_check: Frequency validation mode
        is_series_: Whether input is Series (set during fit)
        index_frequency_: Detected index frequency (set during fit)

    Notes:
        The `frequency` parameter is used for SHIFT ARITHMETIC (period calculation),
        while the DETECTED INDEX FREQUENCY is used for INDEX EXTENSION.

        This allows shifting monthly data on a daily index:
        - Index frequency: 'D' (detected automatically)
        - Shift frequency: 'M' (parameter)
        - Result: Shifts by monthly periods, extends daily index

        Use `frequency_check` to validate consistency between detected and parameter frequencies.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='M')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates, name='GDP')
        >>>
        >>> # Shift forward by 2 monthly periods
        >>> shifter = ShiftTransformer(n_periods=2, frequency='M')
        >>> shifted = shifter.fit_transform(series)
        >>> len(shifted)  # Same length, no data loss
        5
        >>>
        >>> # Works with DataFrames
        >>> df = pd.DataFrame({'GDP': [1, 2, 3], 'CPI': [100, 101, 102]}, index=dates[:3])
        >>> shifter = ShiftTransformer(n_periods=1, frequency='M', frequency_check='warn')
        >>> shifted_df = shifter.fit_transform(df)
        >>>
        >>> # Perfect inverse
        >>> original = shifter.inverse_transform(shifted)
        >>> original.equals(series)
        True
    """

    # Initialisation
    def __init__(
        self,
        n_periods: int,
        frequency: str,
    ):
        """Initialize ShiftTransformer with frequency validation.

        Args:
            n_periods: Number of periods to shift (can be negative)
            frequency: Frequency for period arithmetic ('D', 'M', 'Q', etc.)
        """
        # Initialisation des attributs
        self.n_periods = n_periods
        self.frequency = frequency

    # Méthode d'entraînement
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Fit transformer and detect index frequency.

        Args:
            X: Time series or DataFrame to fit
            y: Ignored

        Returns:
            self

        Raises:
            ValueError: If X doesn't have DatetimeIndex, has non-unique index,
                       or has insufficient observations (< 2)
            ValueError: If frequency_check='raise' and frequencies don't match
        """
        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Validation de la structure temporelle des données
        X = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_frequency(X.index)

        return self

    # Méthode de transformation
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Shift series/dataframe by n_periods with index extension.

        Args:
            X: Time series or DataFrame to transform

        Returns:
            Shifted time series or DataFrame with same type as input

        Raises:
            ValueError: If X is not a pandas Series/DataFrame or doesn't have DatetimeIndex
        """
        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Validation de la structure temporelle des données
        X = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_frequency(X.index)

        # Branchement selon le type de données
        return self._shift_by_periods(data=X, n_periods=self.n_periods)

    # Méthode d'inversion de la transformation
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse shift to recover original data.

        This is stateless - recalculates everything based on inverse logic.

        Args:
            X: Transformed series or DataFrame

        Returns:
            Original series or DataFrame (perfect symmetry with transform)

        Raises:
            ValueError: If X is not a pandas Series/DataFrame or doesn't have DatetimeIndex
        """
        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Validation de la structure temporelle des données
        X = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_frequency(X.index)

        # Branchement selon le type de données (shift opposé)
        return self._shift_by_periods(data=X, n_periods=-self.n_periods,)

    # Méthode auxiliaire de conversion des périodes de shift en périodes d'index
    def _convert_shift_periods_to_index_periods(self, n_periods: int) -> int:
        """Convert shift periods to equivalent index periods.

        When shift frequency is less granular than index frequency (e.g., monthly
        on daily index), calculates how many index periods correspond to n_periods
        in the shift frequency.

        Args:
            n_periods: Number of periods in shift frequency

        Returns:
            Number of equivalent periods in index frequency

        Examples:
            # Shifting by 2 months on daily index: 2 months ≈ 60 days
            >>> self._convert_shift_periods_to_index_periods(2)
            60
        """
        # Normalisation des fréquences
        shift_freq_normalized = normalize_frequency(self.frequency)
        index_freq_normalized = normalize_frequency(self.index_frequency_)

        # Si les fréquences sont identiques, pas de conversion
        if shift_freq_normalized == index_freq_normalized:
            return n_periods

        # Calcul du facteur de conversion via DurationConverter
        converter = DurationConverter()
        try:
            # Exemple: get_conversion_factor('M', 'D') = 30 (1 mois = 30 jours)
            conversion_factor = converter.get_conversion_factor(
                shift_freq_normalized,
                index_freq_normalized
            )
        except ValueError as e:
            raise ValueError(
                f"Cannot convert between frequencies '{self.frequency}' and "
                f"'{self.index_frequency_}': {str(e)}"
            )

        # Application du facteur avec arrondi approprié
        return round(n_periods * conversion_factor)

    # Méthode auxiliaire de décalage des périodes
    def _shift_by_periods(self, data: Union[pd.Series, pd.DataFrame], n_periods: int) -> Union[pd.Series, pd.DataFrame]:
        """Core shift logic using index extension and truncation.

        Positive shift: Extend at start, drop from end
        Negative shift: Extend at end, drop from start
        This avoids NaN introduction.

        Args:
            data: Series or DataFrame to shift
            n_periods: Number of periods to shift (in shift frequency)

        Returns:
            Series or DataFrame with shifted index (same values, different dates)
        """
        # Cas où aucun shift n'est nécessaire
        if n_periods == 0:
            return data.copy()

        # Validation : shift frequency ne doit PAS être plus granulaire que l'index
        if is_higher_frequency(self.frequency, self.index_frequency_):
            raise ValueError(
                f"Shift frequency '{self.frequency}' cannot be more granular "
                f"than index frequency '{self.index_frequency_}'. "
                f"Cannot shift by {self.frequency} on a {self.index_frequency_} index.\n"
                f"Example: You can shift by months ('M') on a daily ('D') index, "
                f"but not by days ('D') on a monthly ('M') index."
            )

        # Conversion des périodes si les fréquences diffèrent
        index_periods = self._convert_shift_periods_to_index_periods(n_periods)

        # Construction de la fréquence complète avec position et suffixe
        full_freq = build_frequency_string(
            self.index_frequency_,
            self.index_position_,
            self.index_suffix_
        )

        if index_periods > 0:
            # Shift positif : extension au début, suppression à la fin
            extended_index = self._extend_index_start(
                data.index,
                abs(index_periods),
                full_freq
            )
            # Conservation seulement des len(series) premières dates
            new_index = extended_index[:len(data)]

        else:  # index_periods < 0
            # Shift négatif : extension à la fin, suppression au début
            extended_index = self._extend_index_end(
                data.index,
                abs(index_periods),
                full_freq
            )
            # Conservation seulement des len(series) dernières dates
            new_index = extended_index[-len(data):]

        # Création de la série avec le nouvel index
        result = data.copy()
        result.index = new_index

        return result


    # Méthode auxiliaire d'extension de l'index au début
    def _extend_index_start(
        self,
        original_index: pd.DatetimeIndex,
        n_periods: int,
        freq: str
    ) -> pd.DatetimeIndex:
        """Extend index backward by adding periods before the first date.

        Used for positive shifts: adds dates before the original index,
        preserving data by shifting the temporal alignment.

        Args:
            original_index: Original DatetimeIndex
            n_periods: Number of periods to add before first date
            freq: Complete frequency string including position/suffix (e.g., 'MS', 'QE-DEC')

        Returns:
            Extended DatetimeIndex with new periods prepended

        Notes:
            freq should be built using _build_complete_frequency_string()
        """
        # Extraction de la première date
        first_date = original_index[0]

        # Génération de n_periods nouvelles dates avant la première date
        new_dates = pd.date_range(
            end=first_date,
            periods=n_periods + 1,  # +1 car end est inclus
            freq=freq
        )[:-1]  # Exclure first_date (déjà dans original)

        # Concaténation
        return new_dates.append(original_index)

    # Méthode auxiliaire d'extension de l'index à la fin
    def _extend_index_end(
        self,
        original_index: pd.DatetimeIndex,
        n_periods: int,
        freq: str
    ) -> pd.DatetimeIndex:
        """Extend index forward by adding periods after the last date.

        Used for negative shifts: adds dates after the original index,
        preserving data by shifting the temporal alignment.

        Args:
            original_index: Original DatetimeIndex
            n_periods: Number of periods to add after last date
            freq: Complete frequency string including position/suffix (e.g., 'MS', 'QE-DEC')

        Returns:
            Extended DatetimeIndex with new periods appended

        Notes:
            freq should be built using _build_complete_frequency_string()
        """
        # Extraction de la dernière date
        last_date = original_index[-1]

        # Génération de n_periods nouvelles dates après la dernière date
        new_dates = pd.date_range(
            start=last_date,
            periods=n_periods + 1, # +1, car start est déjà inclus
            freq=freq
        )[1:]  # Exclure last_date (déjà dans original)

        # Concaténation
        return original_index.append(new_dates)


# Transformer masquant les séries sur un nombre donnée de périodes
class MaskTransformer(BaseEstimator, TransformerMixin):
    """Simple helper to mask N observations per period.

    This is a pure operational transformer with no inference logic.

    Parameters:
        n_obs: Number of observations to mask per period
        mask_frequency: Frequency for period grouping ('D', 'M', 'Q', 'W', 'h', etc.)

    Attributes:
        n_obs: Stored number of observations to mask per period
        mask_frequency: Stored frequency for period grouping
        prediction_date: Stored prediction date
        original_data_: Original data before masking (for inverse_transform)

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> dates = pd.date_range('2024-01-01', periods=90, freq='D')
        >>> series = pd.Series(range(90), index=dates, name='GDP')
        >>>
        >>> # Mask 2 most recent observations per month
        >>> masker = MaskTransformer(
        ...     n_obs=2,
        ...     mask_frequency='M',
        ...     prediction_date=datetime(2024, 3, 31)
        ... )
        >>> masked = masker.fit_transform(series)
        >>>
        >>> # Restore original data
        >>> original = masker.inverse_transform(masked)
    """

    # Initialisation
    def __init__(self, n_obs: int, mask_frequency: str, how: Literal['first', "last"]="last"):
        """Initialize MaskTransformer.

        Args:
            n_obs: Number of observations to mask per period
            mask_frequency: Frequency for period grouping ('D', 'M', 'Q', etc.)
            how: Reference date for masking
        """
        # Initialisation des attributs
        self.n_obs = n_obs
        self.mask_frequency = mask_frequency
        self.how = how
        # Initialisation des données masquées pour les transformations inverses
        self.masked_data_ = None

    # Méthode d'entraînement
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Fit transformer (no-op for MaskTransformer).

        Args:
            X: Time series to fit
            y: Ignored

        Returns:
            self
        """
        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Validation de la structure temporelle des données
        X = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_frequency(X.index)

        return self

    # Méthode de transformation des données
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Mask N most recent observations per period.

        Args:
            X: Time series to transform

        Returns:
            Masked time series

        Raises:
            ValueError: If X is not a pandas Series or doesn't have DatetimeIndex
        """
        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Validation de la structure temporelle des données
        X = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_frequency(X.index)
    
        return self._mask_n_obs_per_period(X)

    # Méthode de transformation inverse des données
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Restore masked values from stored masked data.

        Combines the input X with the masked rows stored during transform,
        then validates and sorts the combined data to maintain temporal structure.

        Args:
            X: Masked series or DataFrame

        Returns:
            Restored series or DataFrame with original values restored for masked rows

        Raises:
            ValueError: If transform has not been called yet
        """
        if self.masked_data_ is None:
            raise ValueError("Must call transform before inverse_transform")

        # Validation du type de données
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        # Si aucune donnée masquée n'a été stockée, retourner X tel quel
        if len(self.masked_data_) == 0:
            return X.copy()

        # Combine X avec les lignes masquées
        # Concaténation
        combined = pd.concat([X, self.masked_data_])
        # Gestion des duplicats en conservant la dernière occurrence (celles de masked_data_ qui viennent en dernier)
        combined = combined[~combined.index.duplicated(keep='last')]

        # Tri des données pour maintenir la structure de série temporelle
        restored = validate_temporal_data(
            data=combined,
            time_col=None,
            panel_cols=None,
            strict=True,
            sort_data=True,
            return_metadata=False
        )

        return restored

    # Méthode auxiliaire de masque du nombre de périodes adapté
    def _mask_n_obs_per_period(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Core masking logic: mask N most recent observations per period.

        Masks the N most recent observations within each period defined by
        mask_frequency, setting masked values to NaN. Stores masked rows in
        self.masked_data_ for later restoration via inverse_transform.

        Args:
            data: Series or DataFrame to mask

        Returns:
            Series or DataFrame with masked observations (NaN values)

        Raises:
            ValueError: If index frequency is not higher than mask frequency
        """
        # Cas où aucun masquage n'est nécessaire
        if self.n_obs == 0:
            # Initialiser masked_data_ avec une structure vide
            if isinstance(data, pd.Series):
                self.masked_data_ = pd.Series(dtype=data.dtype, name=data.name)
            else:
                self.masked_data_ = pd.DataFrame(columns=data.columns)
            return data.copy()

        # Vérification que la fréquence de l'index est strictement supérieure à la fréquence du masque
        if not is_higher_frequency(self.index_frequency_, self.mask_frequency):
            raise ValueError(
                "The index frequency should be strictly higher than the mask frequency."
                f"The index frequency is {self.index_frequency_} and the mask frequency is {self.mask_frequency}"
            )

        # Copie indépendante des données
        masked_data = data.copy()

        # Génération des périodes
        periods = self._generate_periods(
            start_date=data.index.min(),
            end_date=data.index.max(),
            frequency=self.mask_frequency
        )

        # Liste pour collecter les lignes masquées
        masked_rows_list = []

        # Masque dans chaque période
        for period_start, period_end in periods:
            # Filtre des observations dans cette période
            period_mask = (data.index >= period_start) & (data.index < period_end)
            period_obs = data[period_mask]

            # Calcul du nombre de périodes à masquer
            n_to_mask = min(self.n_obs, len(period_obs))

            if (len(period_obs) > 0) & (self.how == 'last'):
                # Masque des n_obs plus récentes
                most_recent_indices = period_obs.index[-n_to_mask:]
                # Stocker les lignes avant de les masquer
                masked_rows_list.append(data.loc[most_recent_indices])
                masked_data.loc[most_recent_indices] = np.nan
            elif (len(period_obs) > 0) & (self.how =='first'):
                # Masque des n_obs les plus anciennes
                oldest_indices = period_obs.index[:n_to_mask]
                # Stocker les lignes avant de les masquer
                masked_rows_list.append(data.loc[oldest_indices])
                masked_data.loc[oldest_indices] = np.nan

        # Combiner toutes les lignes masquées dans self.masked_data_
        if masked_rows_list:
            self.masked_data_ = pd.concat(masked_rows_list)
        else:
            # Aucune ligne masquée
            if isinstance(data, pd.Series):
                self.masked_data_ = pd.Series(dtype=data.dtype, name=data.name)
            else:
                self.masked_data_ = pd.DataFrame(columns=data.columns)

        return masked_data

    # Méthode auxiliaire de génération de période
    def _generate_periods(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        frequency: str
    ) -> List[tuple[datetime, datetime]]:
        """Generate list of (period_start, period_end) tuples.

        Args:
            start_date: Start date for period generation
            end_date: End date for period generation
            frequency: Frequency code for periods

        Returns:
            List of (period_start, period_end) tuples
        """
        # Reconstitution de la fréquence avec position et suffixe
        pandas_freq = build_frequency_string(
            frequency,
            self.index_position_,
            self.index_suffix_
        )

        # Génération des dates de début de période
        period_starts = pd.date_range(
            start=start_date,
            end=end_date,
            freq=pandas_freq
        )

        # Création des tuples (start, end) pour chaque période
        # Initialisation de la liste des périodes
        periods = []
        # Parcours des périodes
        for period_date in period_starts:
            # Extraction des dates de début et de fin de période
            period_start, period_end = get_period_boundaries(period_date, frequency)
            # Ajout du tupe à la liste
            periods.append((period_start, period_end))

        return periods