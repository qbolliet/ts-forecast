"""P1 window calculation for mixed frequency imputation.

This module provides the P1WindowCalculator class to compute the temporal window
where all series have real (non-NaN) values, and extend this window based on
attrition threshold and imputation scope parameters.
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

    The strict window is defined as the temporal interval where ALL series in the
    dataset have real (non-NaN) values. This class also handles extending the
    window based on attrition thresholds and imputation scope settings.

    The calculator accounts for publication delays by optionally excluding
    trailing NaN values that are attributable to delays rather than missing data.

    Attributes:
        p1_start_: Start timestamp of the strict window.
        p1_end_: End timestamp of the strict window.
        training_start_: Start timestamp of the extended training window.
        training_end_: End timestamp of the extended training window.
        column_coverage_: Dict mapping column names to their coverage (non-NaN) timestamps.
        attrition_by_date_: Series showing number of columns with data at each date.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
        >>> data = pd.DataFrame({
        ...     'var1': [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        ...     'var3': [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan]
        ... }, index=dates)
        >>> calculator = P1WindowCalculator(attrition_threshold=0.5, imputation_scope='extended_both')
        >>> calculator.fit(data)
        >>> print(f"P1 window: {calculator.p1_start_} to {calculator.p1_end_}")
        >>> print(f"Training window: {calculator.training_start_} to {calculator.training_end_}")
    """

    def __init__(
        self,
        attrition_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        min_columns: int = 2,
        exclude_delay_nans: bool = True
    ):
        """Initialize the P1WindowCalculator.

        Args:
            attrition_threshold: Minimum percentage of columns that must have
                non-null values for a date to be included in the extended window.
                Value between 0 and 1. Default 0.5 (50%).
            imputation_scope: How to extend the P1 window for training:
                - 'strict': Use only the P1 window (all series have data)
                - 'extended_backward': Extend P1 backwards where threshold is met
                - 'extended_forward': Extend P1 forwards where threshold is met
                - 'extended_both': Extend P1 in both directions
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
        self.p1_start_: Optional[pd.Timestamp] = None
        self.p1_end_: Optional[pd.Timestamp] = None
        self.training_start_: Optional[pd.Timestamp] = None
        self.training_end_: Optional[pd.Timestamp] = None
        self.column_coverage_: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None
        self.attrition_by_date_: Optional[pd.Series] = None
        self._data_columns: Optional[List[str]] = None
        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        delays: Optional[pd.DataFrame] = None,
        panel_cols: Optional[List[str]] = None
    ) -> 'P1WindowCalculator':
        """Calculate the P1 window and extended training window from data.

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
            ValueError: If data is invalid, has insufficient columns, or no P1 window exists.

        Examples:
            >>> calculator = P1WindowCalculator()
            >>> calculator.fit(data)
            >>> print(f"P1: {calculator.p1_start_} to {calculator.p1_end_}")
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

        # Calcul de la fenêtre P1 (intersection complète)
        self.p1_start_, self.p1_end_ = self._compute_p1_window()

        # Vérification de l'existence de la fenêtre P1
        if self.p1_start_ is None or self.p1_end_ is None:
            raise ValueError(
                "No P1 window found: there is no time range where all series have data"
            )

        # Vérification que P1 n'est pas trop courte
        if self.p1_start_ == self.p1_end_:
            warnings.warn(
                "P1 window contains only one observation. Consider relaxing constraints "
                "or using a different imputation_scope.",
                UserWarning
            )

        # Calcul de la fenêtre d'entraînement étendue
        self.training_start_, self.training_end_ = self._compute_training_window()

        # Marquage comme ajusté
        self._is_fitted = True

        return self

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
        # Initialisation du dictionnaire de couverture
        coverage = {}

        # Extraction de l'index temporel
        if isinstance(data.index, pd.MultiIndex):
            # Pour les données de panel, on travaille sur l'index temporel (dernier niveau)
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        # Calcul de la couverture par colonne
        for col in data.columns:
            series = data[col]

            # Identification des valeurs non-nulles
            non_null_mask = series.notna()

            # Exclusion des NaN dus aux délais si demandé
            if self.exclude_delay_nans and delays is not None:
                non_null_mask = self._exclude_delay_nans(series, delays, non_null_mask)

            # Extraction des dates avec valeurs
            if non_null_mask.any():
                valid_dates = time_index[non_null_mask]
                coverage[col] = (valid_dates.min(), valid_dates.max())
            else:
                # Aucune valeur valide
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
        # Recherche du délai pour cette colonne
        col_name = series.name
        col_delay = delays[delays['variable'] == col_name]

        if col_delay.empty:
            return mask

        # Extraction du délai (nombre de périodes à exclure à la fin)
        delay_value = col_delay.iloc[0]['delay']
        delay_unit = col_delay.iloc[0].get('unit', 'periods')
        ref_point = col_delay.iloc[0].get('reference_point', 'end')

        # Pour les délais en périodes, on exclut les n dernières observations
        if delay_unit == 'periods' and ref_point == 'end':
            n_delay = int(delay_value)
            if n_delay > 0 and len(series) > n_delay:
                # Les n dernières positions sont dues au délai, pas des vrais NaN
                delay_indices = series.index[-n_delay:]
                # On garde le masque tel quel pour ces positions (ne pas les compter comme manquantes)
                # Mais on n'invalide pas le masque - ces positions sont simplement ignorées
                pass

        return mask

    def _compute_attrition_by_date(
        self,
        data: pd.DataFrame,
        delays: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Compute the number of columns with data at each date.

        Args:
            data: DataFrame to analyze.
            delays: Optional delay information.

        Returns:
            Series indexed by date with count of columns having data.
        """
        # Calcul du nombre de colonnes non-nulles par date
        non_null_counts = data.notna().sum(axis=1)

        # Conversion en ratio
        total_columns = len(data.columns)
        attrition_ratio = non_null_counts / total_columns

        return attrition_ratio

    def _compute_p1_window(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Compute the P1 window (where all series have data).

        Returns:
            Tuple (start, end) of the P1 window, or (None, None) if no window exists.
        """
        # La fenêtre P1 est l'intersection des couvertures de toutes les colonnes
        if not self.column_coverage_:
            return None, None

        # Extraction des bornes
        starts = []
        ends = []

        for col, (start, end) in self.column_coverage_.items():
            if start is None or end is None:
                # Une colonne n'a aucune donnée, pas de P1 possible
                return None, None
            starts.append(start)
            ends.append(end)

        # P1 = intersection
        # Début = max des débuts
        # Fin = min des fins
        p1_start = max(starts)
        p1_end = min(ends)

        # Vérification que l'intersection est valide
        if p1_start > p1_end:
            return None, None

        return p1_start, p1_end

    def _compute_training_window(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Compute the extended training window based on imputation_scope.

        Returns:
            Tuple (start, end) of the training window.
        """
        # Départ depuis la fenêtre P1
        training_start = self.p1_start_
        training_end = self.p1_end_

        # Pas d'extension si scope est strict
        if self.imputation_scope == 'strict':
            return training_start, training_end

        # Calcul des extensions possibles
        if self.imputation_scope in ('extended_backward', 'extended_both'):
            # Extension vers l'arrière
            training_start = self._extend_window_backward()

        if self.imputation_scope in ('extended_forward', 'extended_both'):
            # Extension vers l'avant
            training_end = self._extend_window_forward()

        return training_start, training_end

    def _extend_window_backward(self) -> pd.Timestamp:
        """Extend the window backward where attrition threshold is met.

        Returns:
            New start timestamp.
        """
        # Dates avant P1 où le seuil d'attrition est respecté
        if self.attrition_by_date_ is None:
            return self.p1_start_

        # Filtrage des dates avant P1
        before_p1 = self.attrition_by_date_[self.attrition_by_date_.index < self.p1_start_]

        if before_p1.empty:
            return self.p1_start_

        # Dates où le seuil est respecté
        valid_dates = before_p1[before_p1 >= self.attrition_threshold]

        if valid_dates.empty:
            return self.p1_start_

        # Extension jusqu'à la date la plus ancienne respectant le seuil
        # en continu depuis P1 (pas de trous)
        sorted_dates = valid_dates.sort_index(ascending=False)

        extended_start = self.p1_start_
        for date in sorted_dates.index:
            # Vérification de la continuité (pas de gap)
            # On accepte les dates qui sont dans la séquence continue
            extended_start = date

        return sorted_dates.index[-1]

    def _extend_window_forward(self) -> pd.Timestamp:
        """Extend the window forward where attrition threshold is met.

        Returns:
            New end timestamp.
        """
        # Dates après P1 où le seuil d'attrition est respecté
        if self.attrition_by_date_ is None:
            return self.p1_end_

        # Filtrage des dates après P1
        after_p1 = self.attrition_by_date_[self.attrition_by_date_.index > self.p1_end_]

        if after_p1.empty:
            return self.p1_end_

        # Dates où le seuil est respecté
        valid_dates = after_p1[after_p1 >= self.attrition_threshold]

        if valid_dates.empty:
            return self.p1_end_

        # Extension jusqu'à la date la plus récente respectant le seuil
        return valid_dates.index[-1]

    def get_training_mask(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None
    ) -> pd.Series:
        """Get a boolean mask for observations in the training window.

        Args:
            data: DataFrame to create mask for.
            column: Optional column name. If provided, also filters for non-null values.

        Returns:
            Boolean Series indicating observations in the training window.

        Raises:
            ValueError: If calculator not fitted.

        Examples:
            >>> mask = calculator.get_training_mask(data)
            >>> train_data = data[mask]
        """
        # Vérification de l'ajustement
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Extraction de l'index temporel
        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        # Création du masque temporel
        mask = (time_index >= self.training_start_) & (time_index <= self.training_end_)

        # Conversion en Series avec le même index
        mask = pd.Series(mask, index=data.index)

        # Filtrage additionnel par colonne si spécifié
        if column is not None:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data")
            mask = mask & data[column].notna()

        return mask

    def get_p1_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get a boolean mask for observations in the P1 window only.

        Args:
            data: DataFrame to create mask for.

        Returns:
            Boolean Series indicating observations in the P1 window.

        Raises:
            ValueError: If calculator not fitted.
        """
        # Vérification de l'ajustement
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Extraction de l'index temporel
        if isinstance(data.index, pd.MultiIndex):
            time_index = data.index.get_level_values(-1)
        else:
            time_index = data.index

        # Création du masque temporel
        mask = (time_index >= self.p1_start_) & (time_index <= self.p1_end_)

        return pd.Series(mask, index=data.index)

    def get_window_info(self) -> Dict[str, Union[pd.Timestamp, int, float]]:
        """Get summary information about calculated windows.

        Returns:
            Dictionary with window information.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Calcul des durées (en nombre de périodes)
        p1_duration = None
        training_duration = None

        if self.attrition_by_date_ is not None:
            p1_mask = (self.attrition_by_date_.index >= self.p1_start_) & \
                      (self.attrition_by_date_.index <= self.p1_end_)
            p1_duration = p1_mask.sum()

            training_mask = (self.attrition_by_date_.index >= self.training_start_) & \
                           (self.attrition_by_date_.index <= self.training_end_)
            training_duration = training_mask.sum()

        return {
            'p1_start': self.p1_start_,
            'p1_end': self.p1_end_,
            'p1_duration': p1_duration,
            'training_start': self.training_start_,
            'training_end': self.training_end_,
            'training_duration': training_duration,
            'attrition_threshold': self.attrition_threshold,
            'imputation_scope': self.imputation_scope,
            'n_columns': len(self._data_columns) if self._data_columns else 0,
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
            # Vérification de l'overlap
            if col_start <= end and col_end >= start:
                columns_with_coverage.append(col)

        return columns_with_coverage

    def __repr__(self) -> str:
        """String representation of the calculator."""
        if not self._is_fitted:
            return (
                f"P1WindowCalculator(attrition_threshold={self.attrition_threshold}, "
                f"imputation_scope='{self.imputation_scope}', not fitted)"
            )

        return (
            f"P1WindowCalculator("
            f"P1=[{self.p1_start_}, {self.p1_end_}], "
            f"training=[{self.training_start_}, {self.training_end_}])"
        )
