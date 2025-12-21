# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, Union, List, Literal, Tuple
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Importation des modules du package
from tsforecast.utils.frequency import to_pandas_freq, normalize_frequency, is_higher_frequency
from tsforecast.utils.duration.converter import DurationConverter
from tsforecast.utils.position import combine_frequency_position
from tsforecast.utils.time import get_period_boundaries
from tsforecast.utils.validation import validate_temporal_data
from ..frequency.detector import detect_frequency


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
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)

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
        full_freq = self._build_complete_frequency_string()

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

    # Méthode auxiliaire de détection de fréquence
    def _detect_index_frequency(self, index: pd.DatetimeIndex) -> Tuple[str, str, str]:
        """Détecte la fréquence d'un DatetimeIndex.

        Args:
            index: DatetimeIndex to analyze

        Returns:
            Pandas frequency code ('D', 'M', etc.)

        Raises:
            ValueError: If frequency cannot be detected
        """
        # Utilisation de detect_frequency sur un dummy Series avec valeurs
        dummy = pd.Series(range(len(index)), index=index, dtype=float)
        freq = detect_frequency(dummy, literal=False)

        # Renvoie une erreur si aucune fréquence n'est détectée
        if freq is None:
            raise ValueError(
                "Could not detect index frequency. "
                "Index may be irregular or have insufficient observations."
            )

        # Séparation de la fréquence de sa position et de son suffixe afin que la fréquence de l'index soit comparable avec celle demandée
        # Initialisation de l'expression régulière
        # Doit matcher : indicateur [S|E] optionnel [-suffixe] optionnel
        match = re.match(r"([A-Z]+?)([SE])?(-(.*?))?$", freq)
        # Extraction des éléments si un appariement est trouvé
        if match:
            freq_ind, position, _, suffix = match.groups()
            return freq_ind, position, suffix
        else :
            raise ValueError(f"Unable to parse frequency, position and suffix in {freq}. Should follow the format : [FREQ][S|E?]-[SUFFIX?]")

    # Méthode auxiliaire de construction de la chaîne de fréquence complète
    def _build_complete_frequency_string(self) -> str:
        """Build complete frequency string including position and suffix.

        Combines frequency, position (S/E), and suffix (e.g., DEC for quarters)
        to create a complete pandas frequency string.

        Returns:
            Complete frequency string (e.g., 'D', 'MS', 'QE-DEC')

        Examples:
            # Daily frequency (no position/suffix)
            >>> self.index_frequency_ = 'D'
            >>> self.index_position_ = None
            >>> self._build_complete_frequency_string()
            'D'

            # Monthly frequency at start
            >>> self.index_frequency_ = 'M'
            >>> self.index_position_ = 'S'
            >>> self.index_suffix_ = None
            >>> self._build_complete_frequency_string()
            'MS'

            # Quarterly frequency at end with December anchor
            >>> self.index_frequency_ = 'Q'
            >>> self.index_position_ = 'E'
            >>> self.index_suffix_ = 'DEC'
            >>> self._build_complete_frequency_string()
            'QE-DEC'
        """
        # Fréquence de base
        base_freq = self.index_frequency_

        # Si pas de position, retour de la fréquence de base
        if self.index_position_ is None:
            return base_freq

        # Combinaison avec la position
        freq_with_position = combine_frequency_position(base_freq, self.index_position_)

        # Ajout du suffixe si présent
        if self.index_suffix_ is not None:
            return f"{freq_with_position}-{self.index_suffix_}"

        return freq_with_position

    # Méthode auxiliaire d'extension de l'index au début
    def _extend_index_start(
        self,
        original_index: pd.DatetimeIndex,
        n_periods: int,
        freq: str
    ) -> pd.DatetimeIndex:
        """Étend l'index au début (pour shift positif).

        Args:
            original_index: Index original
            n_periods: Nombre de périodes à ajouter
            freq: Complete frequency string including position/suffix (e.g., 'MS', 'QE-DEC')

        Returns:
            Index étendu au début

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
        """Étend l'index à la fin (pour shift négatif).

        Args:
            original_index: Index original
            n_periods: Nombre de périodes à ajouter
            freq: Complete frequency string including position/suffix (e.g., 'MS', 'QE-DEC')

        Returns:
            Index étendu à la fin

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
    def __init__(self, n_obs: int, mask_frequency: str, how: Literal['first', "last"]):
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
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)

        return self

    # Méthode de transformation des données
    def transform(self, X: pd.Series) -> pd.Series:
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
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)
    

        return self._mask_n_obs_per_period(X)

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        """Restore masked values from original data.

        Args:
            X: Masked series

        Returns:
            Original series before masking

        Raises:
            ValueError: If transform has not been called yet
        """
        if self.original_data_ is None:
            raise ValueError("Must call transform before inverse_transform")

        return self.original_data_.copy()

    # Méthode auxiliaire de détection de fréquence
    def _detect_index_frequency(self, index: pd.DatetimeIndex) -> Tuple[str, str, str]:
        """Détecte la fréquence d'un DatetimeIndex.

        Args:
            index: DatetimeIndex to analyze

        Returns:
            Pandas frequency code ('D', 'M', etc.)

        Raises:
            ValueError: If frequency cannot be detected
        """
        # Utilisation de detect_frequency sur un dummy Series avec valeurs
        dummy = pd.Series(range(len(index)), index=index, dtype=float)
        freq = detect_frequency(dummy, literal=False)

        # Renvoie une erreur si aucune fréquence n'est détectée
        if freq is None:
            raise ValueError(
                "Could not detect index frequency. "
                "Index may be irregular or have insufficient observations."
            )

        # Séparation de la fréquence de sa position et de son suffixe afin que la fréquence de l'index soit comparable avec celle demandée
        # Initialisation de l'expression régulière
        # Doit matcher : indicateur [S|E] optionnel [-suffixe] optionnel
        match = re.match(r"([A-Z]+?)([SE])?(-(.*?))?$", freq)
        # Extraction des éléments si un appariement est trouvé
        if match:
            freq_ind, position, _, suffix = match.groups()
            return freq_ind, position, suffix
        else :
            raise ValueError(f"Unable to parse frequency, position and suffix in {freq}. Should follow the format : [FREQ][S|E?]-[SUFFIX?]")

    # Méthode auxiliaire de masque du nombre de périodes adapté
    def _mask_n_obs_per_period(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Core masking logic: mask N most recent obs per period.

        Args:
            series: Series to mask

        Returns:
            Series with masked observations
        """
        # Cas où aucun masquage n'est nécessaire
        if self.n_obs == 0:
            return data.copy()

        # Vérification que la fréquence de l'index est strictement supérieure à la fréquence du masque
        if not is_higher_frequency(self.index_frequency_, self.mask_frequency):
            

        masked_series = series.copy()

        # Génération des périodes
        periods = self._generate_periods(
            start_date=series.index.min(),
            end_date=self.prediction_date,
            frequency=self.mask_frequency
        )

        # Masquage dans chaque période
        for period_start, period_end in periods:
            # Observations dans cette période
            period_mask = (series.index >= period_start) & (series.index < period_end)
            period_obs = series[period_mask]

            if len(period_obs) > 0:
                # Masquer les n_obs plus récentes
                n_to_mask = min(self.n_obs, len(period_obs))
                most_recent_indices = period_obs.index[-n_to_mask:]
                masked_series.loc[most_recent_indices] = np.nan

        return masked_series

    def _generate_periods(
        self,
        start_date: pd.Timestamp,
        end_date: datetime,
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
        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(frequency)

        # Génération des dates de début de période
        period_starts = pd.date_range(
            start=start_date,
            end=end_date,
            freq=pandas_freq
        )

        # Création des tuples (start, end) pour chaque période
        periods = []
        for period_date in period_starts:
            period_start, period_end = get_period_boundaries(period_date, frequency)
            periods.append((period_start, period_end))

        return periods