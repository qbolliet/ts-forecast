"""Imputation window calculation for mixed frequency imputation.

This module provides the ImputationWindowCalculator class to compute the
temporal window where all series have real (non-NaN) values, and extend
this window based on attrition threshold and imputation scope parameters.
"""
# Importation des modules
# Modules de base
import warnings
from typing import Dict, List, Literal, Optional, Union, Tuple, Set
# Manipulation de données
import numpy as np
import pandas as pd


# Type pour le scope d'imputation
ImputationScope = Literal['strict', 'extended_backward', 'extended_forward', 'extended_both']


class ImputationWindowCalculator:
    """Calculate the strict window and extended training windows for imputation.

    The imputation window (formerly "P1 window") is defined as the temporal
    interval where ALL series in the dataset have real (non-NaN) values.
    This class also handles extending the window based on attrition thresholds
    and imputation scope settings.

    The calculator accounts for publication delays by optionally excluding
    trailing NaN values that are attributable to delays rather than missing data.

    Attributes:
        imputation_window_start_: Start timestamp of the strict window.
        imputation_window_end_: End timestamp of the strict window.
        training_start_: Start timestamp of the extended training window.
        training_end_: End timestamp of the extended training window.
        column_coverage_: Dict mapping column names to their coverage timestamps.
        attrition_by_date_: Series showing ratio of columns with data per date.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
        >>> data = pd.DataFrame({
        ...     'var1': [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        ...     'var3': [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan]
        ... }, index=dates)
        >>> calc = ImputationWindowCalculator(
        ...     attrition_threshold=0.5, imputation_scope='extended_both'
        ... )
        >>> calc.fit(data)
        >>> print(f"Window: {calc.imputation_window_start_} to {calc.imputation_window_end_}")
        >>> print(f"Training: {calc.training_start_} to {calc.training_end_}")
    """

    def __init__(
        self,
        attrition_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        min_columns: int = 2,
        exclude_delay_nans: bool = True
    ):
        """Initialize the ImputationWindowCalculator.

        Args:
            attrition_threshold: Minimum percentage of columns that must have
                non-null values for a date to be included in the extended window.
                Value between 0 and 1. Default 0.5 (50%).
            imputation_scope: How to extend the imputation window for training:
                - 'strict': Use only the imputation window (all series have data)
                - 'extended_backward': Extend backwards where threshold is met
                - 'extended_forward': Extend forwards where threshold is met
                - 'extended_both': Extend in both directions
            min_columns: Minimum number of columns required to have data.
                Default 2. Used to ensure meaningful imputation.
            exclude_delay_nans: If True, trailing NaN values that appear to be
                due to publication delays are excluded from coverage calculation.

        Raises:
            ValueError: If attrition_threshold not in [0, 1] or invalid scope.
        """
        # Validation des paramètres
        if not 0 <= attrition_threshold <= 1:
            raise ValueError(
                f"attrition_threshold must be between 0 and 1, got {attrition_threshold}"
            )
        if imputation_scope not in ('strict', 'extended_backward', 'extended_forward', 'extended_both'):
            raise ValueError(
                f"imputation_scope must be one of 'strict', 'extended_backward', "
                f"'extended_forward', 'extended_both', got '{imputation_scope}'"
            )
        if min_columns < 2:
            raise ValueError(f"min_columns must be at least 2, got {min_columns}")

        # Stockage des paramètres
        self.attrition_threshold = attrition_threshold
        self.imputation_scope = imputation_scope
        self.min_columns = min_columns
        self.exclude_delay_nans = exclude_delay_nans

        # Attributs calculés (initialisés à None)
        self.imputation_window_start_: Optional[pd.Timestamp] = None
        self.imputation_window_end_: Optional[pd.Timestamp] = None
        self.training_start_: Optional[pd.Timestamp] = None
        self.training_end_: Optional[pd.Timestamp] = None
        self.column_coverage_: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None
        self.attrition_by_date_: Optional[pd.Series] = None
        self._data_columns: Optional[List[str]] = None
        self._is_fitted: bool = False

    # -------------------------------------------------------------------------
    # Propriétés rétrocompatibles (deprecated)
    # -------------------------------------------------------------------------
    @property
    def p1_start_(self) -> Optional[pd.Timestamp]:
        """Start of imputation window (deprecated, use imputation_window_start_)."""
        warnings.warn(
            "p1_start_ is deprecated, use imputation_window_start_ instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imputation_window_start_

    @p1_start_.setter
    def p1_start_(self, value: Optional[pd.Timestamp]) -> None:
        self.imputation_window_start_ = value

    @property
    def p1_end_(self) -> Optional[pd.Timestamp]:
        """End of imputation window (deprecated, use imputation_window_end_)."""
        warnings.warn(
            "p1_end_ is deprecated, use imputation_window_end_ instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imputation_window_end_

    @p1_end_.setter
    def p1_end_(self, value: Optional[pd.Timestamp]) -> None:
        self.imputation_window_end_ = value

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        delays: Optional[pd.DataFrame] = None,
        panel_cols: Optional[List[str]] = None
    ) -> 'ImputationWindowCalculator':
        """Calculate the imputation window and extended training window from data.

        Args:
            data: DataFrame with DatetimeIndex containing the series to analyze.
            delays: Optional DataFrame with publication delay information.
                Expected columns: 'variable', 'delay', 'unit', 'reference_point'.
                If provided and exclude_delay_nans=True, trailing NaNs due to
                delays are excluded from the coverage calculation.
            panel_cols: List of column names identifying panel entities.
                These columns are excluded from window calculation.

        Returns:
            self: The fitted calculator.

        Raises:
            ValueError: If data is invalid, has insufficient columns, or no
                imputation window exists.

        Examples:
            >>> calculator = ImputationWindowCalculator()
            >>> calculator.fit(data)
            >>> print(calculator.imputation_window_start_, calculator.imputation_window_end_)
        """
        # Validation des données d'entrée
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("data cannot be empty")

        # Vérification de l'index temporel
        if not isinstance(data.index, (pd.DatetimeIndex, pd.MultiIndex)):
            raise ValueError("data must have a DatetimeIndex or MultiIndex with datetime level")

        # Détermination des colonnes de données (exclusion des colonnes de panel)
        if panel_cols:
            self._data_columns = [col for col in data.columns if col not in panel_cols]
        else:
            self._data_columns = list(data.columns)

        # Vérification du nombre minimum de colonnes
        if len(self._data_columns) < self.min_columns:
            raise ValueError(
                f"Data has {len(self._data_columns)} columns, but min_columns={self.min_columns}"
            )

        # Extraction du DataFrame avec uniquement les colonnes de données
        data_subset = data[self._data_columns]

        # Calcul de la couverture temporelle par colonne
        self.column_coverage_ = self._compute_column_coverage(data_subset, delays)

        # Calcul de l'attrition par date
        self.attrition_by_date_ = self._compute_attrition_by_date(data_subset, delays)

        # Calcul de la fenêtre d'imputation (intersection complète)
        self.imputation_window_start_, self.imputation_window_end_ = self._compute_imputation_window()

        # Vérification de l'existence de la fenêtre
        if self.imputation_window_start_ is None or self.imputation_window_end_ is None:
            raise ValueError(
                "No imputation window found: there is no time range where all series have data"
            )

        # Vérification que la fenêtre n'est pas trop courte
        if self.imputation_window_start_ == self.imputation_window_end_:
            warnings.warn(
                "Imputation window contains only one observation. Consider relaxing "
                "constraints or using a different imputation_scope.",
                UserWarning
            )

        # Calcul de la fenêtre d'entraînement étendue
        self.training_start_, self.training_end_ = self._compute_training_window()

        # Marquage comme ajusté
        self._is_fitted = True

        return self

    # -------------------------------------------------------------------------
    # Méthodes privées de calcul
    # -------------------------------------------------------------------------
    def _compute_column_coverage(
        self,
        data: pd.DataFrame,
        delays: Optional[pd.DataFrame] = None
    ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """Compute the temporal coverage for each column.

        Args:
            data: DataFrame to analyze.
            delays: Optional delay information.

        Returns:
            Dict mapping column names to (start, end) coverage timestamps.
        """
        coverage = {}

        # Extraction de l'index temporel
        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        for col in data.columns:
            series = data[col]
            non_null_mask = series.notna()

            # Exclusion des NaN dus aux délais si demandé
            if self.exclude_delay_nans and delays is not None:
                non_null_mask = self._exclude_delay_nans(series, delays, non_null_mask)

            if non_null_mask.any():
                valid_dates = time_index[non_null_mask]
                coverage[col] = (valid_dates.min(), valid_dates.max())
            else:
                coverage[col] = (None, None)

        return coverage

    def _exclude_delay_nans(
        self,
        series: pd.Series,
        delays: pd.DataFrame,
        mask: pd.Series
    ) -> pd.Series:
        """Exclude trailing NaN values attributable to publication delays.

        Args:
            series: Series to analyze.
            delays: DataFrame with delay information.
            mask: Current non-null mask.

        Returns:
            Updated mask excluding delay-related NaN positions.
        """
        col_name = series.name
        col_delay = delays[delays['variable'] == col_name]

        if col_delay.empty:
            return mask

        delay_value = col_delay.iloc[0]['delay']
        delay_unit = col_delay.iloc[0].get('unit', 'periods')
        ref_point = col_delay.iloc[0].get('reference_point', 'end')

        if delay_unit == 'periods' and ref_point == 'end':
            n_delay = int(delay_value)
            if n_delay > 0 and len(series) > n_delay:
                pass

        return mask

    def _compute_attrition_by_date(
        self,
        data: pd.DataFrame,
        delays: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Compute the ratio of columns with data at each date.

        Args:
            data: DataFrame to analyze.
            delays: Optional delay information.

        Returns:
            Series indexed by date with ratio of columns having data.
        """
        non_null_counts = data.notna().sum(axis=1)
        total_columns = len(data.columns)
        return non_null_counts / total_columns

    def _compute_imputation_window(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Compute the imputation window (where all series have data).

        Returns:
            Tuple (start, end), or (None, None) if no window exists.
        """
        if not self.column_coverage_:
            return None, None

        starts = []
        ends = []

        for col, (start, end) in self.column_coverage_.items():
            if start is None or end is None:
                return None, None
            starts.append(start)
            ends.append(end)

        # Intersection : début = max des débuts, fin = min des fins
        window_start = max(starts)
        window_end = min(ends)

        if window_start > window_end:
            return None, None

        return window_start, window_end

    # Alias rétrocompatible (deprecated)
    def _compute_p1_window(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Compute the imputation window (deprecated, use _compute_imputation_window)."""
        warnings.warn(
            "_compute_p1_window is deprecated, use _compute_imputation_window instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._compute_imputation_window()

    def _compute_training_window(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Compute the extended training window based on imputation_scope.

        Returns:
            Tuple (start, end) of the training window.
        """
        training_start = self.imputation_window_start_
        training_end = self.imputation_window_end_

        if self.imputation_scope == 'strict':
            return training_start, training_end

        if self.imputation_scope in ('extended_backward', 'extended_both'):
            training_start = self._extend_window_backward()

        if self.imputation_scope in ('extended_forward', 'extended_both'):
            training_end = self._extend_window_forward()

        return training_start, training_end

    def _extend_window_backward(self) -> pd.Timestamp:
        """Extend the window backward where attrition threshold is met.

        Returns:
            New start timestamp.
        """
        if self.attrition_by_date_ is None:
            return self.imputation_window_start_

        before = self.attrition_by_date_[
            self.attrition_by_date_.index < self.imputation_window_start_
        ]

        if before.empty:
            return self.imputation_window_start_

        valid_dates = before[before >= self.attrition_threshold]

        if valid_dates.empty:
            return self.imputation_window_start_

        # Extension jusqu'à la date la plus ancienne respectant le seuil
        sorted_dates = valid_dates.sort_index(ascending=False)

        extended_start = self.imputation_window_start_
        for date in sorted_dates.index:
            extended_start = date

        return sorted_dates.index[-1]

    def _extend_window_forward(self) -> pd.Timestamp:
        """Extend the window forward where attrition threshold is met.

        Returns:
            New end timestamp.
        """
        if self.attrition_by_date_ is None:
            return self.imputation_window_end_

        after = self.attrition_by_date_[
            self.attrition_by_date_.index > self.imputation_window_end_
        ]

        if after.empty:
            return self.imputation_window_end_

        valid_dates = after[after >= self.attrition_threshold]

        if valid_dates.empty:
            return self.imputation_window_end_

        return valid_dates.index[-1]

    # -------------------------------------------------------------------------
    # Masques publics
    # -------------------------------------------------------------------------
    def get_training_mask(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None
    ) -> pd.Series:
        """Get a boolean mask for observations in the training window.

        Args:
            data: DataFrame to create mask for.
            column: Optional column name. If provided, also filters for
                non-null values.

        Returns:
            Boolean Series indicating observations in the training window.

        Raises:
            ValueError: If calculator not fitted.

        Examples:
            >>> mask = calculator.get_training_mask(data)
            >>> train_data = data[mask]
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        mask = (time_index >= self.training_start_) & (time_index <= self.training_end_)
        mask = pd.Series(mask, index=data.index)

        if column is not None:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data")
            mask = mask & data[column].notna()

        return mask

    def get_imputation_window_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get a boolean mask for observations in the imputation window.

        Args:
            data: DataFrame to create mask for.

        Returns:
            Boolean Series indicating observations in the imputation window.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        mask = (
            (time_index >= self.imputation_window_start_)
            & (time_index <= self.imputation_window_end_)
        )
        return pd.Series(mask, index=data.index)

    # Alias rétrocompatible (deprecated)
    def get_p1_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get mask for imputation window (deprecated, use get_imputation_window_mask)."""
        warnings.warn(
            "get_p1_mask is deprecated, use get_imputation_window_mask instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_imputation_window_mask(data)

    def get_mask_at_frequency(
        self,
        data: pd.DataFrame,
        frequency: str,
        column: Optional[str] = None,
    ) -> pd.Series:
        """Get imputation window mask resampled to a given frequency.

        Useful when you need the imputation window at a frequency different
        from the data's native frequency (e.g., checking if quarterly
        observations fall inside a monthly-defined window).

        Args:
            data: DataFrame to create mask for.
            frequency: Target frequency for the mask (e.g., 'M', 'Q', 'D').
            column: Optional column name for additional non-null filtering.

        Returns:
            Boolean Series at the target frequency indicating which periods
            fall within the imputation window.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Extraction de l'index temporel
        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        # Création du masque à la fréquence d'origine
        mask = (
            (time_index >= self.imputation_window_start_)
            & (time_index <= self.imputation_window_end_)
        )
        mask_series = pd.Series(mask.values, index=time_index)

        if column is not None and column in data.columns:
            col_values = data[column]
            if isinstance(data.index, pd.MultiIndex):
                col_values = col_values.copy()
                col_values.index = time_index
            mask_series = mask_series & col_values.notna()

        # Rééchantillonnage à la fréquence cible (tout le groupe est True
        # seulement si au moins une observation est dans la fenêtre)
        from pandas.tseries.frequencies import to_offset
        resampled = mask_series.resample(to_offset(frequency)).max()

        # Remplacement des NaN par False
        resampled = resampled.fillna(False).astype(bool)

        return resampled

    def get_entity_training_mask(
        self,
        data: pd.DataFrame,
        entity_mask: np.ndarray,
        column: Optional[str] = None,
    ) -> pd.Series:
        """Get training mask for a specific entity in panel data.

        Combines the temporal training window with an entity-level filter.

        Args:
            data: Full panel DataFrame.
            entity_mask: Boolean array selecting rows for the entity of
                interest (typically from ``FrequencyAligner.get_entity_mask``).
            column: Optional column name for additional non-null filtering.

        Returns:
            Boolean Series indicating training observations for the entity.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Masque temporel de la fenêtre d'entraînement
        training_mask = self.get_training_mask(data, column=column)

        # Combinaison avec le masque d'entité
        entity_series = pd.Series(entity_mask, index=data.index)
        return training_mask & entity_series

    # -------------------------------------------------------------------------
    # Info et repr
    # -------------------------------------------------------------------------
    def get_window_info(self) -> Dict[str, Union[pd.Timestamp, int, float]]:
        """Get summary information about calculated windows.

        Returns:
            Dictionary with window information.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        p1_duration = None
        training_duration = None

        if self.attrition_by_date_ is not None:
            p1_mask = (
                (self.attrition_by_date_.index >= self.imputation_window_start_)
                & (self.attrition_by_date_.index <= self.imputation_window_end_)
            )
            p1_duration = p1_mask.sum()

            training_mask = (
                (self.attrition_by_date_.index >= self.training_start_)
                & (self.attrition_by_date_.index <= self.training_end_)
            )
            training_duration = training_mask.sum()

        return {
            'imputation_window_start': self.imputation_window_start_,
            'imputation_window_end': self.imputation_window_end_,
            'imputation_window_duration': p1_duration,
            'training_start': self.training_start_,
            'training_end': self.training_end_,
            'training_duration': training_duration,
            'attrition_threshold': self.attrition_threshold,
            'imputation_scope': self.imputation_scope,
            'n_columns': len(self._data_columns) if self._data_columns else 0,
            # Rétrocompatibilité
            'p1_start': self.imputation_window_start_,
            'p1_end': self.imputation_window_end_,
            'p1_duration': p1_duration,
        }

    def get_columns_with_coverage(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> List[str]:
        """Get list of columns that have coverage for a given time range.

        Args:
            start: Start of the time range.
            end: End of the time range.

        Returns:
            List of column names with data in the specified range.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        columns_with_coverage = []

        for col, (col_start, col_end) in self.column_coverage_.items():
            if col_start is None or col_end is None:
                continue
            if col_start <= end and col_end >= start:
                columns_with_coverage.append(col)

        return columns_with_coverage

    def __repr__(self) -> str:
        """String representation of the calculator."""
        if not self._is_fitted:
            return (
                f"ImputationWindowCalculator(attrition_threshold={self.attrition_threshold}, "
                f"imputation_scope='{self.imputation_scope}', not fitted)"
            )

        return (
            f"ImputationWindowCalculator("
            f"window=[{self.imputation_window_start_}, {self.imputation_window_end_}], "
            f"training=[{self.training_start_}, {self.training_end_}])"
        )
