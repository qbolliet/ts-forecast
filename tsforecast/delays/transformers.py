"""Sklearn-compatible transformers for applying publication delays.

This module provides sklearn API-compatible transformers for applying
publication delays to time series and panel data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Callable
from datetime import datetime, timedelta
import warnings
from sklearn.utils.validation import check_is_fitted
import copy

from ..utils.base_transformers import PanelTimeSeriesTransformer, ReversibleTransformerMixin
from .delay_calculator import ReleaseDelayCalculator


class ReleaseDelayTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
    """Sklearn transformer for applying publication delays.

    This class applies publication delays to data using either
    a provided delay dictionary or delays calculated from a delay calculator.

    Two transformation modes available:
    - 'shift': Shifts data according to publication delays
    - 'mask': Masks observations that would not yet be available

    Parameters:
        delays_dict (Optional[Dict]): Dictionary of delays by indicator/entity
        delay_calculator (Optional[ReleaseDelayCalculator]): Delay calculator
        mode (str): Transformation mode ('shift' or 'mask')
        prediction_date (Union[str, datetime]): Reference prediction date
        reference_point (str): Reference point for delays ('start' or 'end')
        time_col (str): Name of time column
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions
        handle_missing_delays (str): Strategy for missing delays ('ignore', 'warn', 'error')
        default_delay (Optional[float]): Default delay if missing
        validate_input (bool): Validate input data

    Attributes:
        delays_dict_ (Dict): Dictionary of delays used
        mode_ (str): Transformation mode
        prediction_date_ (datetime): Reference prediction date
        reference_point_ (str): Reference point
        original_data_ (Optional[pd.DataFrame]): Original data for inverse_transform
        transformation_metadata_ (Dict): Transformation metadata

    Examples:
        >>> from tsforecast.delays import ReleaseDelayTransformer
        >>>
        >>> # Use with delay dictionary
        >>> delays = {'GDP': 45.0, 'inflation': 30.0}
        >>> transformer = ReleaseDelayTransformer(
        ...     delays_dict=delays,
        ...     mode='shift',
        ...     prediction_date='2023-12-01'
        ... )
        >>>
        >>> # Apply transformation
        >>> X_transformed = transformer.fit_transform(X)
        >>>
        >>> # Reverse transformation
        >>> X_original = transformer.inverse_transform(X_transformed)
        >>>
        >>> # Use with delay calculator
        >>> calculator = ReleaseDelayCalculator(delay_data=df)
        >>> transformer = ReleaseDelayTransformer(
        ...     delay_calculator=calculator,
        ...     mode='mask',
        ...     prediction_date='today'
        ... )
    """

    def __init__(self,
                 delays_dict: Optional[Dict[Union[str, Tuple[str, str]], float]] = None,
                 delay_calculator: Optional[ReleaseDelayCalculator] = None,
                 mode: str = 'shift',
                 prediction_date: Union[str, datetime] = 'today',
                 reference_point: str = 'end',
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 handle_missing_delays: str = 'warn',
                 default_delay: Optional[float] = None,
                 validate_input: bool = True):
        """Initialize the publication delay transformer.

        Args:
            delays_dict: Dictionary of delays by indicator
            delay_calculator: Calculator to get delays from data
            mode: Transformation mode ('shift' or 'mask')
            prediction_date: Reference date ('today' or datetime)
            reference_point: Reference point ('start' or 'end')
            time_col: Name of time column
            panel_cols: Panel dimension columns
            handle_missing_delays: Missing delay handling
            default_delay: Default delay if missing
            validate_input: Input data validation
        """
        # Initialisation de la classe parent
        super().__init__(time_col=time_col, panel_cols=panel_cols, validate_input=validate_input)

        # Validation des paramètres
        if delays_dict is None and delay_calculator is None:
            raise ValueError("You must provide either delays_dict or delay_calculator")

        if mode not in ['shift', 'mask']:
            raise ValueError("mode must be 'shift' or 'mask'")

        if reference_point not in ['start', 'end']:
            raise ValueError("reference_point must be 'start' or 'end'")

        if handle_missing_delays not in ['ignore', 'warn', 'error']:
            raise ValueError("handle_missing_delays must be 'ignore', 'warn' or 'error'")

        # Assignation des paramètres
        self.delays_dict = delays_dict
        self.delay_calculator = delay_calculator
        self.mode = mode
        self.prediction_date = prediction_date
        self.reference_point = reference_point
        self.handle_missing_delays = handle_missing_delays
        self.default_delay = default_delay

        # Attributs à définir lors du fit
        self.delays_dict_ = None
        self.mode_ = mode
        self.prediction_date_ = None
        self.reference_point_ = reference_point
        self.original_data_ = None
        self.transformation_metadata_ = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Transformer fitting (internal method).

        Args:
            X: Input data
            y: Target variable (not used)
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = self._resolve_prediction_date(self.prediction_date, X)

        # Obtention du dictionnaire des délais
        if self.delays_dict is not None:
            self.delays_dict_ = self.delays_dict.copy()
        elif self.delay_calculator is not None:
            self.delays_dict_ = self._get_delays_from_calculator(X)
        else:
            raise RuntimeError("No delay source available")

        # Stockage des métadonnées de transformation
        self.transformation_metadata_ = {
            'fitted_on': datetime.utcnow(),
            'mode': self.mode_,
            'prediction_date': self.prediction_date_,
            'reference_point': self.reference_point_,
            'delays_count': len(self.delays_dict_),
            'original_shape': X.shape,
            'original_columns': list(X.columns)
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation (internal method).

        Args:
            X: Data to transform

        Returns:
            Data transformed according to chosen mode
        """
        # Stockage des données originales pour inverse_transform
        self.original_data_ = X.copy()

        # Application de la transformation selon le mode
        if self.mode_ == 'shift':
            return self._apply_shift_transformation(X)
        elif self.mode_ == 'mask':
            return self._apply_mask_transformation(X)
        else:
            raise ValueError(f"Unknown transformation mode: {self.mode_}")

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse applied transformation.

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original format

        Raises:
            ValueError: If transformation cannot be reversed
        """
        check_is_fitted(self, 'delays_dict_')

        if self.original_data_ is None:
            raise ValueError("No original data stored for inversion")

        if self.mode_ == 'shift':
            return self._reverse_shift_transformation(X)
        elif self.mode_ == 'mask':
            return self._reverse_mask_transformation(X)
        else:
            raise ValueError(f"Unknown transformation mode for inversion: {self.mode_}")

    def _resolve_prediction_date(self,
                                prediction_date: Union[str, datetime],
                                X: pd.DataFrame) -> datetime:
        """Resolve prediction date from provided parameter.

        Args:
            prediction_date: Prediction date ('today' or datetime)
            X: Input data to extract latest date if needed

        Returns:
            Resolved prediction date
        """
        if isinstance(prediction_date, str):
            if prediction_date.lower() == 'today':
                return datetime.now()
            else:
                try:
                    return pd.to_datetime(prediction_date).to_pydatetime()
                except:
                    raise ValueError(f"Invalid date format: {prediction_date}")
        elif isinstance(prediction_date, datetime):
            return prediction_date
        else:
            raise ValueError("prediction_date must be 'today', a string or datetime")

    def _get_delays_from_calculator(self, X: pd.DataFrame) -> Dict[Union[str, Tuple[str, str]], float]:
        """Get delays from delay calculator.

        Args:
            X: Input data to determine indicators

        Returns:
            Dictionary of delays by indicator/entity
        """
        # Identification des indicateurs dans les données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Calcul des délais
        group_by_entity = self.is_panel_
        delays = self.delay_calculator.calculate_median_delays(
            reference_point=self.reference_point_,
            group_by_entity=group_by_entity,
            indicators=data_cols
        )

        return delays

    def _apply_shift_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation in 'shift' mode.

        In shift mode, data is shifted towards the future according to their
        publication delays. For example, if GDP has a 45-day delay, GDP values
        are shifted 45 days into the future.

        Args:
            X: Data to transform

        Returns:
            Data with shifted values
        """
        X_shifted = X.copy()

        # Identification des colonnes de données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Application du décalage pour chaque colonne
        for col in data_cols:
            delay = self._get_delay_for_column(col, X)

            if delay is None:
                continue

            if self.is_panel_:
                # Traitement par groupe pour les données panel
                X_shifted = self._shift_column_panel(X_shifted, col, delay)
            else:
                # Traitement pour série temporelle simple
                X_shifted = self._shift_column_timeseries(X_shifted, col, delay)

        return X_shifted

    def _apply_mask_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation in 'mask' mode.

        In mask mode, observations that would not yet be available
        at the prediction date are replaced by NaN.

        Args:
            X: Data to transform

        Returns:
            Data with future values masked
        """
        X_masked = X.copy()

        # Identification des colonnes de données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Application du masquage pour chaque colonne
        for col in data_cols:
            delay = self._get_delay_for_column(col, X)

            if delay is None:
                continue

            if self.is_panel_:
                # Traitement par groupe pour les données panel
                X_masked = self._mask_column_panel(X_masked, col, delay)
            else:
                # Traitement pour série temporelle simple
                X_masked = self._mask_column_timeseries(X_masked, col, delay)

        return X_masked

    def _get_delay_for_column(self,
                             column: str,
                             X: pd.DataFrame) -> Optional[float]:
        """Get delay for a given column.

        Args:
            column: Column name
            X: Data (for panel context if needed)

        Returns:
            Delay in days or None if not found
        """
        delay = None

        if self.is_panel_:
            # Pour données panel, chercher avec les entités
            # Note: cette implémentation suppose une seule entité par DataFrame
            # Pour plusieurs entités, il faudrait grouper par entité
            if self.panel_cols:
                # Extraction de la première entité comme exemple
                first_row = X.iloc[0]
                entity_parts = []
                for pcol in self.panel_cols:
                    if pcol in first_row:
                        entity_parts.append(str(first_row[pcol]))
                entity_key = '|'.join(entity_parts) if entity_parts else None

                # Tentative avec clé entité-indicateur
                if entity_key:
                    delay = self.delays_dict_.get((entity_key, column))

                # Fallback sur indicateur seul
                if delay is None:
                    delay = self.delays_dict_.get(column)
        else:
            # Pour série temporelle simple
            delay = self.delays_dict_.get(column)

        # Gestion des délais manquants
        if delay is None:
            if self.handle_missing_delays == 'error':
                raise ValueError(f"Missing delay for column '{column}'")
            elif self.handle_missing_delays == 'warn':
                warnings.warn(f"Missing delay for column '{column}', using default delay")

            delay = self.default_delay

        return delay

    def _shift_column_timeseries(self,
                                X: pd.DataFrame,
                                column: str,
                                delay_days: float) -> pd.DataFrame:
        """Décale une colonne pour série temporelle simple.

        Args:
            X: DataFrame à modifier
            column: Nom de la colonne à décaler
            delay_days: Délai en jours

        Returns:
            DataFrame avec la colonne décalée
        """
        if delay_days == 0:
            return X

        # Création d'un index temporel si nécessaire
        if self.time_col in X.columns:
            time_index = pd.to_datetime(X[self.time_col])
        else:
            time_index = pd.to_datetime(X.index)

        # Calcul de la nouvelle position temporelle
        shifted_dates = time_index + pd.Timedelta(days=delay_days)

        # Création d'un DataFrame temporaire pour le réalignement
        temp_df = pd.DataFrame({
            'shifted_date': shifted_dates,
            'original_value': X[column]
        })

        # Réindexation sur les dates originales
        if self.time_col in X.columns:
            original_dates = time_index
        else:
            original_dates = time_index

        # Interpolation/alignement des valeurs décalées
        # Les valeurs sont décalées vers le futur, donc certaines positions
        # deviennent NaN (données pas encore disponibles)
        X_copy = X.copy()

        # Reset des valeurs pour cette colonne
        X_copy[column] = np.nan

        # Réassignation des valeurs aux nouvelles positions
        for i, (orig_date, new_date, value) in enumerate(zip(time_index, shifted_dates, X[column])):
            if pd.notna(value):
                # Trouve la position la plus proche dans l'index original
                closest_idx = (abs(time_index - new_date)).argmin()
                if abs((time_index.iloc[closest_idx] - new_date).days) <= 1:  # Tolérance d'1 jour
                    X_copy.iloc[closest_idx, X_copy.columns.get_loc(column)] = value

        return X_copy

    def _shift_column_panel(self,
                           X: pd.DataFrame,
                           column: str,
                           delay_days: float) -> pd.DataFrame:
        """Décale une colonne pour données panel.

        Args:
            X: DataFrame à modifier
            column: Nom de la colonne à décaler
            delay_days: Délai en jours

        Returns:
            DataFrame avec la colonne décalée
        """
        if delay_days == 0:
            return X

        X_copy = X.copy()

        # Traitement par groupe panel
        for group_values, group_data in X.groupby(self.panel_cols):
            group_indices = group_data.index

            # Application du décalage pour ce groupe
            shifted_group = self._shift_column_timeseries(
                group_data, column, delay_days
            )

            # Réassignation des valeurs
            X_copy.loc[group_indices, column] = shifted_group[column]

        return X_copy

    def _mask_column_timeseries(self,
                               X: pd.DataFrame,
                               column: str,
                               delay_days: float) -> pd.DataFrame:
        """Masque une colonne pour série temporelle simple.

        Args:
            X: DataFrame à modifier
            column: Nom de la colonne à masquer
            delay_days: Délai en jours

        Returns:
            DataFrame avec la colonne masquée
        """
        if delay_days <= 0:
            return X

        X_copy = X.copy()

        # Calcul de la date limite de disponibilité
        cutoff_date = self.prediction_date_ - timedelta(days=delay_days)

        # Identification des indices temporels
        if self.time_col in X.columns:
            time_index = pd.to_datetime(X[self.time_col])
        else:
            time_index = pd.to_datetime(X.index)

        # Masquage des observations trop récentes
        mask = time_index > cutoff_date
        X_copy.loc[mask, column] = np.nan

        return X_copy

    def _mask_column_panel(self,
                          X: pd.DataFrame,
                          column: str,
                          delay_days: float) -> pd.DataFrame:
        """Masque une colonne pour données panel.

        Args:
            X: DataFrame à modifier
            column: Nom de la colonne à masquer
            delay_days: Délai en jours

        Returns:
            DataFrame avec la colonne masquée
        """
        if delay_days <= 0:
            return X

        X_copy = X.copy()

        # Traitement par groupe panel
        for group_values, group_data in X.groupby(self.panel_cols):
            group_indices = group_data.index

            # Application du masquage pour ce groupe
            masked_group = self._mask_column_timeseries(
                group_data, column, delay_days
            )

            # Réassignation des valeurs
            X_copy.loc[group_indices, column] = masked_group[column]

        return X_copy

    def _reverse_shift_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse shift transformation.

        Args:
            X: Transformed data

        Returns:
            Reconstructed original data
        """
        if self.original_data_ is None:
            raise ValueError("Original data not available for inversion")

        # Pour la transformation shift, on retourne les données originales
        # car le shift est difficile à inverser parfaitement sans perte d'information
        warnings.warn("Inversion of 'shift' transformation returns original data")
        return self.original_data_.copy()

    def _reverse_mask_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse mask transformation.

        Args:
            X: Transformed data

        Returns:
            Reconstructed original data
        """
        if self.original_data_ is None:
            raise ValueError("Original data not available for inversion")

        # Pour la transformation mask, on remplace les NaN par les valeurs originales
        X_restored = X.copy()

        # Identification des colonnes de données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Restauration des valeurs masquées
        for col in data_cols:
            if col in self.original_data_.columns:
                mask = X_restored[col].isna()
                X_restored.loc[mask, col] = self.original_data_.loc[mask, col]

        return X_restored

    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformation.

        Returns:
            Dictionary containing transformation information
        """
        check_is_fitted(self, 'delays_dict_')

        return {
            'mode': self.mode_,
            'prediction_date': self.prediction_date_.isoformat() if self.prediction_date_ else None,
            'reference_point': self.reference_point_,
            'delays_applied': dict(self.delays_dict_),
            'transformation_metadata': self.transformation_metadata_,
            'is_panel_data': self.is_panel_,
            'time_column': self.time_col,
            'panel_columns': self.panel_cols
        }

    def update_delays(self, new_delays: Dict[Union[str, Tuple[str, str]], float]) -> None:
        """Update delays dictionary.

        Args:
            new_delays: New delays dictionary

        Raises:
            RuntimeError: If transformer has not been fitted
        """
        check_is_fitted(self, 'delays_dict_')

        self.delays_dict_.update(new_delays)

        # Mise à jour des métadonnées
        self.transformation_metadata_['delays_updated'] = datetime.utcnow()
        self.transformation_metadata_['delays_count'] = len(self.delays_dict_)

    def get_available_delays(self) -> Dict[Union[str, Tuple[str, str]], float]:
        """Return dictionary of currently available delays.

        Returns:
            Dictionary of delays by indicator/entity
        """
        if self.delays_dict_ is None:
            return {}
        return self.delays_dict_.copy()


def create_delay_transformer_from_calculator(
    calculator: ReleaseDelayCalculator,
    mode: str = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    **kwargs) -> ReleaseDelayTransformer:
    """Factory function to create transformer from calculator.

    Args:
        calculator: Configured delay calculator
        mode: Transformation mode ('shift' or 'mask')
        prediction_date: Reference prediction date
        **kwargs: Additional arguments for ReleaseDelayTransformer

    Returns:
        Configured ReleaseDelayTransformer instance

    Examples:
        >>> from tsforecast.delays import ReleaseDelayCalculator
        >>> from tsforecast.delays import create_delay_transformer_from_calculator
        >>>
        >>> delay_df = pd.DataFrame({...})
        >>> calculator = ReleaseDelayCalculator(delay_data=delay_df)
        >>> transformer = create_delay_transformer_from_calculator(
        ...     calculator, mode='mask', prediction_date='2023-12-01'
        ... )
    """
    return ReleaseDelayTransformer(
        delay_calculator=calculator,
        mode=mode,
        prediction_date=prediction_date,
        **kwargs
    )


def create_delay_transformer_from_dict(
    delays_dict: Dict[Union[str, Tuple[str, str]], float],
    mode: str = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    **kwargs) -> ReleaseDelayTransformer:
    """Factory function to create transformer from dictionary.

    Args:
        delays_dict: Dictionary of delays by indicator
        mode: Transformation mode ('shift' or 'mask')
        prediction_date: Reference prediction date
        **kwargs: Additional arguments for ReleaseDelayTransformer

    Returns:
        Configured ReleaseDelayTransformer instance

    Examples:
        >>> delays = {'GDP': 45.0, 'inflation': 30.0, 'unemployment': 15.0}
        >>> transformer = create_delay_transformer_from_dict(
        ...     delays, mode='shift', prediction_date='2023-12-01'
        ... )
    """
    return ReleaseDelayTransformer(
        delays_dict=delays_dict,
        mode=mode,
        prediction_date=prediction_date,
        **kwargs
    )