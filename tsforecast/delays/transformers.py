"""Sklearn-compatible transformers for applying publication delays.

This module provides sklearn API-compatible transformers for applying
publication delays to time series and panel data.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Callable
from datetime import datetime, timedelta
import warnings
# Sklearn
from sklearn.utils.validation import check_is_fitted

# Importation des modules du package
from ..utils.time import resolve_date
from ..utils.base_transformers import PanelTimeSeriesTransformer, ReversibleTransformerMixin


# Classe d'application des délais de publication
class PublicationDelayTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
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

    # Initialisation
    def __init__(self,
                 delays: Union[Dict[Union[str, tuple], float], pd.DataFrame],
                 delays_unit: Optional[str] = None,
                 mode: str = 'shift',
                 prediction_date: Union[str, datetime] = 'today',
                 prediction_date_format: Optional[str] = None,
                 reference_point: str = 'end',
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 handle_missing_delays: str = 'warn',
                 default_delay: Optional[float] = None,
                 validate_input: bool = True,
                 strict_validation: bool = True,
                 auto_sort: bool = False):
        """Initialize the publication delay transformer.

        Args:
            delays: Dictionary or DataFrame of delays by indicator
            delays_unit: Unit of delays ('D' for days, 's' for seconds, etc.).
                        If None, inferred from DataFrame 'unit' column or defaults to 'D'
            mode: Transformation mode ('shift' or 'mask')
            prediction_date: Reference date ('today' or datetime)
            reference_point: Reference point ('start' or 'end')
            time_col: Name of time column
            panel_cols: Panel dimension columns
            handle_missing_delays: Missing delay handling
            default_delay: Default delay if missing
            validate_input: Input data validation
            strict_validation: If True, raises errors; if False, emits warnings
            auto_sort: If True, automatically sorts unsorted data
        """
        # Initialisation de la classe parent
        super().__init__(
            time_col=time_col,
            panel_cols=panel_cols,
            validate_input=validate_input,
            strict_validation=strict_validation,
            auto_sort=auto_sort
        )

        # Validation des paramètres
        # Validation du mode d'application des délais de publication
        if mode not in ['shift', 'mask']:
            raise ValueError("'mode' must be 'shift' or 'mask'")
        # Validation du point de référence des délais de publication
        if reference_point not in ['start', 'end']:
            raise ValueError("'reference_point' must be 'start' or 'end'")
        # Validation de la gestion des délais manquants
        if handle_missing_delays not in ['ignore', 'warn', 'error']:
            raise ValueError("'handle_missing_delays' must be 'ignore', 'warn' or 'error'")

        # Validation de la structure du paramètre delays
        _validate_delays_parameter(delays, panel_cols)

        # Inférence de l'unité des délais si non fournie
        if delays_unit is None:
            if isinstance(delays, pd.DataFrame):
                # Inférence depuis le DataFrame si colonne 'unit' présente
                if 'unit' in delays.columns:
                    # Une seule unité par transformer (utilise la première valeur)
                    self.delays_unit = delays['unit'].iloc[0]
                else:
                    # Défaut si colonne 'unit' absente
                    self.delays_unit = 'D'
            else:
                # Défaut pour les dictionnaires
                self.delays_unit = 'D'
        else:
            # Utilisation de la valeur explicite
            self.delays_unit = delays_unit

        # Assignation des paramètres
        self.delays = delays
        self.mode = mode
        self.prediction_date = prediction_date
        self.prediction_date_format = prediction_date_format
        self.reference_point = reference_point
        self.handle_missing_delays = handle_missing_delays
        self.default_delay = default_delay

        # Attributs à définir lors du fit
        self.delays_series_ = None
        self.mode_ = mode
        self.prediction_date_ = None
        self.reference_point_ = reference_point
        self.original_data_ = None
        self.transformation_metadata_ = {}

    # Méthode d'entraînement du transformer
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Transformer fitting (internal method).

        Args:
            X: Input data
            y: Target variable (not used)
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(date=self.prediction_date, format=self.prediction_date_format)

        # Création de la Series des délais
        if self.delays is not None:
            if isinstance(self.delays, pd.DataFrame):
                # Conversion DataFrame -> Series
                self.delays_series_ = self.delays['applicable_delay']

            elif isinstance(self.delays, dict):
                # Conversion directe Dict -> Series
                self.delays_series_ = pd.Series(self.delays)

                # Nommage des niveaux de l'index
                if isinstance(self.delays_series_.index, pd.MultiIndex):
                    # Détermination du nombre de niveaux d'entités
                    n_levels = self.delays_series_.index.nlevels
                    # Noms des niveaux : entity_1, ..., entity_N, indicator
                    level_names = [f'entity_{i}' for i in range(1, n_levels)] + ['indicator']
                    self.delays_series_.index.names = level_names
                else:
                    # Index simple : nom 'indicator'
                    self.delays_series_.index.name = 'indicator'
            else:
                raise TypeError(
                    f"delays must be Dict or DataFrame, got {type(self.delays)}"
                )
        else:
            raise RuntimeError("No delay source available")

        # Stockage des métadonnées de transformation
        self.transformation_metadata_ = {
            'mode': self.mode_,
            'prediction_date': self.prediction_date_,
            'reference_point': self.reference_point_,
            'delays_count': len(self.delays_series_),
            'original_shape': X.shape,
            'original_columns': list(X.columns)
        }

    # Fonction de transformation des données
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

    # Méthode de transformation inverse des données
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse applied transformation.

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original format

        Raises:
            ValueError: If transformation cannot be reversed
        """
        # Vérification que le transformer est estimé
        check_is_fitted(self, 'delays_series_')

        # Vérification que les données originales sont renseignées
        if self.original_data_ is None:
            raise ValueError("No original data stored for inversion")

        # Distinction suivant le mode de transformation
        if self.mode_ == 'shift':
            X_restored = self._reverse_shift_transformation(X)
        elif self.mode_ == 'mask':
            X_restored = self._reverse_mask_transformation(X)
        else:
            raise ValueError(f"Unknown transformation mode for inversion: {self.mode_}")

        # Restauration de la structure originale si conversion appliquée
        X_restored = self._restore_structure_if_converted(X_restored)

        return X_restored

    # Méthode d'application que la transformation par décalage
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
        # Copie indépendante du jeu de données
        X_shifted = X.copy()

        # Identification des colonnes de données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Application du décalage pour chaque colonne
        for col in data_cols:
            # Extraction du délai
            delay = self._get_delay_for_column(col, X)
            # Si aucun délai n'est identifié pour l'indicateur, passe à la suivante
            if delay is None:
                continue
            
            if self.is_panel_:
                # Traitement par groupe pour les données panel
                X_shifted = self._shift_column_panel(X_shifted, col, delay)
            else:
                # Traitement pour série temporelle simple
                X_shifted = self._shift_column_timeseries(X_shifted, col, delay)

        return X_shifted

    # Méthode d'application de la transformation par masque
    def _apply_mask_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation in 'mask' mode.

        In mask mode, observations that would not yet be available
        at the prediction date are replaced by NaN.

        Args:
            X: Data to transform

        Returns:
            Data with future values masked
        """
        # Copie indépendante des données
        X_masked = X.copy()

        # Identification des colonnes de données
        data_cols = [col for col in X.columns
                    if col != self.time_col and (not self.panel_cols or col not in self.panel_cols)]

        # Application du masque pour chaque colonne
        for col in data_cols:
            # Identification du délai de publication
            delay = self._get_delay_for_column(col, X)
            # Si aucun délai n'est identifié pour l'indicateur, passe à la suivante
            if delay is None:
                continue

            if self.is_panel_:
                # Traitement par groupe pour les données panel
                X_masked = self._mask_column_panel(X_masked, col, delay)
            else:
                # Traitement pour série temporelle simple
                X_masked = self._mask_column_timeseries(X_masked, col, delay)

        return X_masked

    # Méthode auxiliaire du délai valable
    def _get_delay_for_column(self,
                             column: str) -> Optional[Union[float, pd.Series]]:
        """Get delay for a given column.

        Recherche le délai de publication pour une colonne donnée, avec support
        des structures multi-niveaux et fallback sur l'indicateur seul.

        Args:
            column: Column name (indicator)

        Returns:
            Delay value or None if not found
        """
        # Initialisation du délai
        delay = None

        if self.is_panel_ and self.panel_cols:
            # Extraction des valeurs d'entité et de leur délai associé
            try:
                delay = self.delays_series_.xs(key=column, level='indicator')
            except KeyError:
                delay = None
        else:
            # Série temporelle simple : lookup direct sur l'indicateur
            try:
                delay = self.delays_series_.loc[column]
            except KeyError:
                delay = None

        # Gestion des délais manquants
        if delay is None:
            if self.handle_missing_delays == 'error':
                raise ValueError(f"Missing delay for column '{column}'")
            elif self.handle_missing_delays == 'warn':
                warnings.warn(f"Missing delay for column '{column}', using default delay")
            # Application du délai par défaut
            delay = self.default_delay

        return delay

    # Méthode auxiliaire de shift pour les séries temporelles
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
        check_is_fitted(self, 'delays_series_')

        return {
            'mode': self.mode_,
            'prediction_date': self.prediction_date_.isoformat() if self.prediction_date_ else None,
            'reference_point': self.reference_point_,
            'delays_applied': self.delays_series_.to_dict(),
            'transformation_metadata': self.transformation_metadata_,
            'is_panel_data': self.is_panel_,
            'time_column': self.time_col,
            'panel_columns': self.panel_cols
        }

    def update_delays(self, new_delays: Union[Dict[Union[str, tuple], float], pd.Series]) -> None:
        """Update delays series.

        Accepte soit un dictionnaire, soit une Series pandas pour mettre à jour
        les délais existants.

        Args:
            new_delays: New delays (Dict or pandas.Series)

        Raises:
            RuntimeError: If transformer has not been fitted
        """
        check_is_fitted(self, 'delays_series_')

        # Conversion en Series si Dict fourni
        if isinstance(new_delays, dict):
            new_delays_series = pd.Series(new_delays)
        elif isinstance(new_delays, pd.Series):
            new_delays_series = new_delays
        else:
            raise TypeError(f"new_delays must be Dict or Series, got {type(new_delays)}")

        # Fusion avec la Series existante (update)
        # pd.Series.update() modifie en place, donc on utilise combine_first
        self.delays_series_ = new_delays_series.combine_first(self.delays_series_)

        # Mise à jour des métadonnées
        self.transformation_metadata_['delays_updated'] = datetime.utcnow()
        self.transformation_metadata_['delays_count'] = len(self.delays_series_)

    def get_available_delays(self) -> pd.Series:
        """Return Series of currently available delays.

        BREAKING CHANGE: Retourne maintenant une pandas.Series au lieu d'un Dict.
        Utiliser `.to_dict()` si un dictionnaire est nécessaire.

        Returns:
            Series of delays with Index (time series) or MultiIndex (panel data)

        Examples:
            >>> transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'inflation': 30.0})
            >>> transformer.fit(X)
            >>> delays_series = transformer.get_available_delays()
            >>> # Pour obtenir un Dict :
            >>> delays_dict = delays_series.to_dict()
        """
        if self.delays_series_ is None:
            return pd.Series(dtype=float)
        return self.delays_series_.copy()

    def get_delays_as_dataframe(self) -> pd.DataFrame:
        """Return delays in DataFrame format.

        Retourne les délais au format DataFrame compatible avec
        calculate_applicable_delay().

        Returns:
            DataFrame with columns: applicable_delay, unit
            Index: indicator or MultiIndex(entity_1, ..., entity_N, indicator)

        Raises:
            RuntimeError: If transformer has not been fitted yet

        Examples:
            >>> # Initialize with dictionary
            >>> transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'inflation': 30.0})
            >>> transformer.fit(X)
            >>> df = transformer.get_delays_as_dataframe()
            >>> print(df.columns)
            Index(['applicable_delay', 'unit'], dtype='object')
            >>>
            >>> # Panel data example (2-level)
            >>> delays = {('France', 'GDP'): 45.0, ('Germany', 'GDP'): 38.0}
            >>> transformer = PublicationDelayTransformer(delays=delays, panel_cols=['country'])
            >>> transformer.fit(X)
            >>> df = transformer.get_delays_as_dataframe()
            >>> print(df.index.names)
            ['entity_1', 'indicator']
            >>>
            >>> # Panel data example (multi-level)
            >>> delays = {('France', 'Paris', 'GDP'): 45.0, ('France', 'Lyon', 'GDP'): 48.0}
            >>> transformer = PublicationDelayTransformer(delays=delays, panel_cols=['country', 'city'])
            >>> transformer.fit(X)
            >>> df = transformer.get_delays_as_dataframe()
            >>> print(df.index.names)
            ['entity_1', 'entity_2', 'indicator']
        """
        if self.delays_series_ is None:
            raise RuntimeError("Transformer has not been fitted yet")

        return delays_dict_to_dataframe(
            self.delays_series_.to_dict(),
            unit=self.delays_unit
        )


def create_delay_transformer_from_calculator(
    calculator: Any,  # ReleaseDelayCalculator type hint temporarily disabled
    mode: str = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    **kwargs) -> 'PublicationDelayTransformer':
    """Factory function to create transformer from calculator.

    Note: This function uses calculate_applicable_delay from the calculator
    to get delays as a DataFrame, which is then passed to the transformer.

    Args:
        calculator: Configured delay calculator with calculate_applicable_delay() method
        mode: Transformation mode ('shift' or 'mask')
        prediction_date: Reference prediction date
        **kwargs: Additional arguments for PublicationDelayTransformer

    Returns:
        Configured PublicationDelayTransformer instance

    Examples:
        >>> # from tsforecast.delays import ReleaseDelayCalculator
        >>> # from tsforecast.delays import create_delay_transformer_from_calculator
        >>>
        >>> # delay_df = pd.DataFrame({...})
        >>> # calculator = ReleaseDelayCalculator(delay_data=delay_df)
        >>> # transformer = create_delay_transformer_from_calculator(
        >>> #     calculator, mode='mask', prediction_date='2023-12-01'
        >>> # )
        >>> pass  # Placeholder until ReleaseDelayCalculator is implemented
    """
    # Obtention des délais depuis le calculator au format DataFrame
    delays_df = calculator.calculate_applicable_delay()

    return PublicationDelayTransformer(
        delays=delays_df,
        mode=mode,
        prediction_date=prediction_date,
        **kwargs
    )


def create_delay_transformer_from_dict(
    delays_dict: Dict[Union[str, Tuple[str, str]], float],
    mode: str = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    **kwargs) -> 'PublicationDelayTransformer':
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
    return PublicationDelayTransformer(
        delays=delays_dict,
        mode=mode,
        prediction_date=prediction_date,
        **kwargs
    )


# Fonctions utilitaires de conversion des formats de délais
# Fonction de convertion d'un dictionnaire de délais en DataFrame
def delays_dict_to_dataframe(
    delays_dict: Dict[Union[str, tuple], float],
    unit: str = 'D'
) -> pd.DataFrame:
    """Convert delays dictionary to DataFrame format.

    Convertit un dictionnaire de délais en DataFrame compatible avec
    le format retourné par calculate_applicable_delay(). Supporte les tuples
    multi-niveaux où les n-1 premiers éléments sont des entités et le dernier
    est l'indicateur.

    Args:
        delays_dict: Dictionary of delays {indicator: delay} or {(*entities, indicator): delay}
                    where tuple can have 2+ elements (entity1, ..., entityN, indicator)
        unit: Unit of delays ('D' for days, 's' for seconds, 'us' for microseconds)

    Returns:
        DataFrame with columns: applicable_delay, unit
        Index: 'indicator' for time series, or MultiIndex(['entity_1', ..., 'entity_N', 'indicator']) for panel

    Examples:
        >>> # Simple time series format
        >>> delays = {'GDP': 45.0, 'inflation': 30.0}
        >>> df = delays_dict_to_dataframe(delays)
        >>>
        >>> # Panel data format (2-level)
        >>> panel_delays = {('France', 'GDP'): 45.0, ('Germany', 'GDP'): 38.0}
        >>> df = delays_dict_to_dataframe(panel_delays)
        >>>
        >>> # Panel data format (multi-level)
        >>> multi_delays = {('France', 'Paris', 'GDP'): 45.0, ('France', 'Lyon', 'GDP'): 48.0}
        >>> df = delays_dict_to_dataframe(multi_delays)
    """
    # Détection du format (time series vs panel)
    is_panel = any(isinstance(k, tuple) for k in delays_dict.keys())

    if is_panel:
        # Détermination du nombre de niveaux d'entités
        # On cherche la longueur maximale des tuples pour déterminer la structure
        max_tuple_len = max(len(k) if isinstance(k, tuple) else 1 for k in delays_dict.keys())

        # Vérification de la cohérence : tous les tuples doivent avoir la même longueur
        tuple_lengths = [len(k) if isinstance(k, tuple) else 1 for k in delays_dict.keys()]
        if len(set(tuple_lengths)) > 1:
            raise ValueError(
                f"Inconsistent tuple lengths in delays_dict: {set(tuple_lengths)}. "
                "All tuples must have the same length (n-1 entity levels + 1 indicator)."
            )

        # Format panel: extraction des entités et indicateurs
        # Pour un tuple de longueur N : éléments [0:N-1] = entités, élément [N-1] = indicateur
        n_entity_levels = max_tuple_len - 1 if max_tuple_len > 1 else 0

        data = []
        for key, delay in delays_dict.items():
            if isinstance(key, tuple) and len(key) >= 2:
                # Extraction des niveaux d'entités et de l'indicateur
                entity_values = key[:-1]  # Tous sauf le dernier
                indicator = key[-1]  # Le dernier élément

                row = {}
                for i, entity_val in enumerate(entity_values, start=1):
                    row[f'entity_{i}'] = entity_val
                row['indicator'] = indicator
                row['applicable_delay'] = delay
                row['unit'] = unit
                data.append(row)
            else:
                # Clé non-tuple ou tuple à 1 élément dans un contexte panel
                # On suppose que c'est un indicateur seul
                row = {}
                for i in range(1, n_entity_levels + 1):
                    row[f'entity_{i}'] = None
                row['indicator'] = key if not isinstance(key, tuple) else key[0]
                row['applicable_delay'] = delay
                row['unit'] = unit
                data.append(row)

        # Création du DataFrame
        df = pd.DataFrame(data)

        # Création du MultiIndex si toutes les entités sont non-null
        entity_cols = [f'entity_{i}' for i in range(1, n_entity_levels + 1)]
        if all(df[col].notna().all() for col in entity_cols):
            # Toutes les entités sont présentes : créer MultiIndex
            index_cols = entity_cols + ['indicator']
            df = df.set_index(index_cols)
        else:
            # Certaines entités manquent : utiliser seulement 'indicator' comme index
            df = df.set_index('indicator')
            df = df.drop(columns=entity_cols)
    else:
        # Format time series simple
        data = []
        for indicator, delay in delays_dict.items():
            data.append({
                'indicator': indicator,
                'applicable_delay': delay,
                'unit': unit
            })
        df = pd.DataFrame(data)
        df = df.set_index('indicator')

    return df


# Fonction auxiliaire de validation des paramètres de délais de publication
def _validate_delays_parameter(
    delays: Union[Dict, pd.DataFrame],
    panel_cols: Optional[List[str]] = None
) -> None:
    """Validate delays parameter structure.

    Valide la structure du paramètre delays selon qu'il s'agit
    d'un dictionnaire ou d'un DataFrame.

    Args:
        delays: Dictionary or DataFrame of delays
        panel_cols: Panel columns to determine expected format

    Raises:
        ValueError: If structure is invalid
        TypeError: If type is neither Dict nor DataFrame
    """
    # Détection de la structure de panel
    is_panel = panel_cols is not None and len(panel_cols) > 0

    # Distinction suivant le type de l'argument
    if isinstance(delays, dict):
        # Validation du dictionnaire
        if len(delays) == 0:
            warnings.warn("Empty delays dictionary provided")
            return

        # Vérification de la cohérence des clés
        first_key = next(iter(delays.keys()))
        is_tuple_key = isinstance(first_key, tuple)

        # Pour panel data, les clés devraient être des tuples
        if is_panel and not is_tuple_key:
            warnings.warn(
                "Panel data detected but delays keys are not tuples. "
                "Expected format: {(entity, indicator): delay}"
            )

        # Vérification que toutes les valeurs sont numériques
        for key, value in delays.items():
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"Delay value for {key} must be numeric, got {type(value)}"
                )

    elif isinstance(delays, pd.DataFrame):
        # Validation du DataFrame
        if 'applicable_delay' not in delays.columns:
            raise ValueError(
                "DataFrame must contain 'applicable_delay' column. "
                "Expected format from calculate_applicable_delay()"
            )

        # Vérification que l'index est approprié
        if is_panel and not isinstance(delays.index, pd.MultiIndex):
            warnings.warn(
                "Panel data detected but DataFrame index is not MultiIndex. "
                "Expected MultiIndex with (entity, indicator)"
            )

        # Vérification des valeurs numériques
        if not pd.api.types.is_numeric_dtype(delays['applicable_delay']):
            raise ValueError(
                "'applicable_delay' column must contain numeric values"
            )

    else:
        raise TypeError(
            f"delays must be Dict or DataFrame, got {type(delays)}"
        )