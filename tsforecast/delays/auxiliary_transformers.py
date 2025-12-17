# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, Union, List, Literal
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Importation des modules du package
from tsforecast.utils.frequency import to_pandas_freq, normalize_frequency
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
        frequency_check: Literal["ignore", "warn", "raise"] = "ignore"
    ):
        """Initialize ShiftTransformer with frequency validation.

        Args:
            n_periods: Number of periods to shift (can be negative)
            frequency: Frequency for period arithmetic ('D', 'M', 'Q', etc.)
            frequency_check: Frequency validation mode ('ignore', 'warn', 'raise')
        """
        # Initialisation des attributs
        self.n_periods = n_periods
        self.frequency = frequency
        self.frequency_check = frequency_check

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
        
        # Stockage du type de données
        self.is_series_ = isinstance(X, pd.Series)

        # Détection de la fréquence de l'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = self._detect_index_frequency(X.index)

        # Validation de la fréquence si demandé
        if self.frequency_check != "ignore":
            self._validate_frequency()

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

        # Validation de la fréquence si demandé
        if self.frequency_check != "ignore":
            self._validate_frequency()

        # Branchement selon le type de données
        if self.is_series_:
            return self._shift_by_periods(X, self.n_periods, self.frequency)
        else:
            # Pour DataFrame : appliquer le shift à toutes les colonnes
            result = X.copy()
            for col in result.columns:
                result[col] = self._shift_by_periods(result[col], self.n_periods, self.frequency)
            return result

    # Méthode d'inversion de la transformation
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse shift to recover original data.

        This is STATELESS - recalculates everything based on inverse logic.

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

        # Vérification que l'index est un DateTime
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        # Branchement selon le type de données (shift opposé)
        if self.is_series_:
            return self._shift_by_periods(X, -self.n_periods, self.frequency)
        else:
            # Pour DataFrame : appliquer le shift inverse à toutes les colonnes
            result = X.copy()
            for col in result.columns:
                result[col] = self._shift_by_periods(result[col], -self.n_periods, self.frequency)
            return result

    # Méthode auxiliaire de décalage des périodes
    def _shift_by_periods(self, series: pd.Series, n_periods: int, frequency: str) -> pd.Series:
        """Core shift logic - shifts the index by n periods, not the values.

        The values stay at the same array positions, but their associated dates change.
        Preserves the exact position of each timestamp within its period.

        Args:
            series: Series to shift
            n_periods: Number of periods to shift
            frequency: Frequency for period arithmetic

        Returns:
            Series with shifted index (same values, different dates)
        """
        # Cas où aucun shift n'est nécessaire
        if n_periods == 0:
            return series.copy()

        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(frequency)

        # Conversion de l'index en PeriodIndex pour l'arithmétique de périodes
        period_index = series.index.to_period(freq=pandas_freq)

        # Shift des périodes
        shifted_periods = period_index + n_periods

        # Pour préserver la position exacte des timestamps dans chaque période,
        # on calcule le décalage entre le timestamp original et le début de sa période
        shifted_index_list = []
        for i, (original_ts, original_period, shifted_period) in enumerate(
            zip(series.index, period_index, shifted_periods)
        ):
            # Début de la période originale
            original_period_start = original_period.to_timestamp()
            # Décalage entre le timestamp original et le début de sa période
            offset = original_ts - original_period_start
            # Appliquer le même décalage à la nouvelle période
            shifted_period_start = shifted_period.to_timestamp()
            shifted_ts = shifted_period_start + offset
            shifted_index_list.append(shifted_ts)

        # Création du nouvel index
        shifted_index = pd.DatetimeIndex(shifted_index_list)

        # Création de la série avec le nouvel index
        result = pd.Series(series.values, index=shifted_index, name=series.name)

        return result

    # Méthode auxiliaire de détection de fréquence
    def _detect_index_frequency(self, index: pd.DatetimeIndex) -> str:
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
        match = re.match(r"([A-Z]+)([SE])?(-(.*?))?$", freq)
        # Extraction des éléments si un appariement est trouvé
        if match:
            freq_ind, position, _, suffix = match.groups()
            return freq_ind, position, suffix
        else :
            raise ValueError(f"Unable to parse frequency, position and suffix in {freq}. Should follow the format : [FREQ][S|E?]-[SUFFIX?]")

    

    # Méthode auxiliaire de validation de fréquence
    def _validate_frequency(self):
        """Valide la cohérence entre fréquence détectée et fréquence spécifiée.

        Raises:
            ValueError: If frequency_check='raise' and frequencies don't match
        """
        # Normalisation des deux fréquences pour comparaison
        try:
            detected_normalized = normalize_frequency(self.index_frequency_)
            param_normalized = normalize_frequency(self.frequency)
        except (ValueError, KeyError):
            # Si normalisation échoue, utiliser les valeurs brutes
            detected_normalized = self.index_frequency_
            param_normalized = self.frequency

        # Gestion des cas d'erreur
        if detected_normalized != param_normalized:
            message = (
                f"Frequency validation failed:\n"
                f"  Detected index frequency: {self.index_frequency_}\n"
                f"  Parameter frequency: {self.frequency}\n"
                f"  Hint: Set frequency_check='ignore' to bypass validation, "
                f"or use frequency='{self.index_frequency_}' to match detected frequency."
            )

            if self.frequency_check == "raise":
                raise ValueError(message)
            elif self.frequency_check == "warn":
                warnings.warn(message, UserWarning)

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
            freq: Fréquence détectée de l'index

        Returns:
            Index étendu au début
        """
        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(freq)
        # Extraction de la première date
        first_date = original_index[0]

        # Génération de n_periods nouvelles dates avant la première date
        new_dates = pd.date_range(
            end=first_date,
            periods=n_periods + 1,  # +1 car end est inclus
            freq=pandas_freq
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
            freq: Fréquence détectée de l'index

        Returns:
            Index étendu à la fin
        """
        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(freq)
        # Extraction de la dernière date
        last_date = original_index[-1]

        # Génération de n_periods nouvelles dates après la dernière date
        new_dates = pd.date_range(
            start=last_date,
            periods=n_periods + 1, # +1, car start est déjà inclus
            freq=pandas_freq
        )[1:]  # Exclure last_date (déjà dans original)

        # Concaténation
        return original_index.append(new_dates)


class MaskTransformer(BaseEstimator, TransformerMixin):
    """Simple helper to mask N observations per period.

    This is a pure operational transformer with no inference logic.
    Panel data handling is done by the orchestrator (PublicationDelayTransformer).

    Parameters:
        n_obs: Number of observations to mask per period
        mask_frequency: Frequency for period grouping ('D', 'M', 'Q', 'W', 'h', etc.)
        prediction_date: Reference date for masking

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

    def __init__(self, n_obs: int, mask_frequency: str, prediction_date: datetime):
        """Initialize MaskTransformer.

        Args:
            n_obs: Number of observations to mask per period
            mask_frequency: Frequency for period grouping ('D', 'M', 'Q', etc.)
            prediction_date: Reference date for masking
        """
        self.n_obs = n_obs
        self.mask_frequency = mask_frequency
        self.prediction_date = prediction_date
        self.original_data_ = None

    def fit(self, X: pd.Series, y=None):
        """Fit transformer (no-op for MaskTransformer).

        Args:
            X: Time series to fit
            y: Ignored

        Returns:
            self
        """
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Mask N most recent observations per period.

        Args:
            X: Time series to transform

        Returns:
            Masked time series

        Raises:
            ValueError: If X is not a pandas Series or doesn't have DatetimeIndex
        """
        # Validation de l'entrée
        if not isinstance(X, pd.Series):
            raise ValueError("X must be a pandas Series")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        # Stockage des données originales pour inverse_transform
        self.original_data_ = X.copy()

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

    def _mask_n_obs_per_period(self, series: pd.Series) -> pd.Series:
        """Core masking logic: mask N most recent obs per period.

        Args:
            series: Series to mask

        Returns:
            Series with masked observations
        """
        # Cas où aucun masquage n'est nécessaire
        if self.n_obs == 0:
            return series.copy()

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