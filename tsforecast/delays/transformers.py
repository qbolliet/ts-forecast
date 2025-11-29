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



# Fonctions utilitaires de conversion des formats de délais
# Fonction de convertion d'un dictionnaire de délais en DataFrame
def delays_dict_to_dataframe(
    delays_dict: Dict[Union[str, Tuple[str, str]], float],
    unit: str = 'D'
) -> pd.DataFrame:
    """Convert delays dictionary to DataFrame format.

    Convertit un dictionnaire de délais en DataFrame compatible avec
    le format retourné par calculate_applicable_delay().

    Args:
        delays_dict: Dictionary of delays {indicator: delay} or {(entity, indicator): delay}
        unit: Unit of delays ('D' for days, 's' for seconds, 'us' for microseconds)

    Returns:
        DataFrame with columns: applicable_delay, unit, indicator (and entity if panel)

    Examples:
        >>> # Simple time series format
        >>> delays = {'GDP': 45.0, 'inflation': 30.0}
        >>> df = delays_dict_to_dataframe(delays)
        >>>
        >>> # Panel data format
        >>> panel_delays = {('France', 'GDP'): 45.0, ('Germany', 'GDP'): 38.0}
        >>> df = delays_dict_to_dataframe(panel_delays)
    """
    # Détection du format (time series vs panel)
    is_panel = any(isinstance(k, tuple) for k in delays_dict.keys())

    if is_panel:
        # Format panel: extraction des entités et indicateurs
        data = []
        for key, delay in delays_dict.items():
            if isinstance(key, tuple):
                entity, indicator = key
                data.append({
                    'entity': entity,
                    'indicator': indicator,
                    'applicable_delay': delay,
                    'unit': unit
                })
            else:
                # Clé non-tuple dans un contexte panel (on suppose que c'est un indicateur seul)
                data.append({
                    'entity': None,
                    'indicator': key,
                    'applicable_delay': delay,
                    'unit': unit
                })
        # Création du jeu de données
        df = pd.DataFrame(data)
        # Création du MultiIndex uniquement si toutes les entités sont non-null
        if df['entity'].notna().all():
            df = df.set_index(['entity', 'indicator'])
        else:
            df = df.set_index('indicator')
            df = df.drop(columns=['entity'])
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

# Conversion d'un DataFrame en dictionnaire de délais de publication
def delays_dataframe_to_dict(
    delays_df: pd.DataFrame,
    use_entity: bool = None
) -> Dict[Union[str, Tuple[str, str]], float]:
    """Convert delays DataFrame to dictionary format.

    Convertit un DataFrame de délais (format calculate_applicable_delay)
    en dictionnaire compatible avec PublicationDelayTransformer.

    Args:
        delays_df: DataFrame with 'applicable_delay' column and index on indicator
                   (and optionally entity)
        use_entity: If True, uses (entity, indicator) format. If None, auto-detected.

    Returns:
        Dictionary {indicator: delay} or {(entity, indicator): delay}

    Raises:
        ValueError: If 'applicable_delay' column is missing

    Examples:
        >>> # DataFrame from calculate_applicable_delay
        >>> df = pd.DataFrame({'applicable_delay': [45.0, 30.0]},
        ...                   index=pd.Index(['GDP', 'inflation'], name='indicator'))
        >>> delays_dict = delays_dataframe_to_dict(df)
        >>> # Returns: {'GDP': 45.0, 'inflation': 30.0}
    """
    # Validation de la présence de la colonne applicable_delay
    if 'applicable_delay' not in delays_df.columns:
        raise ValueError(
            "DataFrame must contain 'applicable_delay' column. "
            "Expected format from calculate_applicable_delay()"
        )

    # Détection automatique du format si use_entity est None
    if use_entity is None:
        use_entity = isinstance(delays_df.index, pd.MultiIndex)

    # Initialisation du dictionnaire résultat
    delays_dict = {}

    if use_entity and isinstance(delays_df.index, pd.MultiIndex):
        # Format panel: (entity, indicator) -> delay
        for idx, row in delays_df.iterrows():
            # MultiIndex: idx est un tuple (entity, indicator, ...)
            delays_dict[idx] = row['applicable_delay']
    else:
        # Format time series: indicator -> delay
        for idx, row in delays_df.iterrows():
            delays_dict[idx] = row['applicable_delay']

    return delays_dict

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
            raise ValueError("mode must be 'shift' or 'mask'")
        # Validation du point de référence des délais de publication
        if reference_point not in ['start', 'end']:
            raise ValueError("reference_point must be 'start' or 'end'")
        # Validation de la gestion des délais manquants
        if handle_missing_delays not in ['ignore', 'warn', 'error']:
            raise ValueError("handle_missing_delays must be 'ignore', 'warn' or 'error'")

        # Validation de la structure du paramètre delays
        _validate_delays_parameter(delays, panel_cols)

        # Assignation des paramètres
        self.delays = delays
        self.mode = mode
        self.prediction_date = prediction_date
        self.prediction_date_format = prediction_date_format
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

    # Méthode d'entraînement du transformer
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Transformer fitting (internal method).

        Args:
            X: Input data
            y: Target variable (not used)
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(date=self.prediction_date, format=self.prediction_date_format)

        # Obtention du dictionnaire des délais
        if self.delays is not None:
            # Conversion en dictionnaire si nécessaire
            if isinstance(self.delays, pd.DataFrame):
                # Conversion DataFrame -> Dict avec préservation de l'unité
                self.delays_dict_ = delays_dataframe_to_dict(
                    self.delays,
                    use_entity=self.is_panel_
                )
                # Stockage de l'unité pour get_delays_as_dataframe()
                if 'unit' in self.delays.columns:
                    self._delays_unit = self.delays['unit'].iloc[0]
                else:
                    self._delays_unit = 'D'
            elif isinstance(self.delays, dict):
                # Copie directe du dictionnaire
                self.delays_dict_ = self.delays.copy()
                self._delays_unit = 'D'  # Défaut pour dicts
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
            'delays_count': len(self.delays_dict_),
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
        check_is_fitted(self, 'delays_dict_')

        # Vérification que les données originales sont renseignées
        if self.original_data_ is None:
            raise ValueError("No original data stored for inversion")

        # Distinction suivant le mode de transformation
        if self.mode_ == 'shift':
            return self._reverse_shift_transformation(X)
        elif self.mode_ == 'mask':
            return self._reverse_mask_transformation(X)
        else:
            raise ValueError(f"Unknown transformation mode for inversion: {self.mode_}")

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
                             column: str,
                             X: pd.DataFrame) -> Optional[float]:
        """Get delay for a given column.

        Args:
            column: Column name
            X: Data (for panel context if needed)

        Returns:
            Delay in days or None if not found
        """
        # Initialisation du délai
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
            # Application du délai par défaut
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

    def get_delays_as_dataframe(self) -> pd.DataFrame:
        """Return delays in DataFrame format.

        Retourne les délais au format DataFrame compatible avec
        calculate_applicable_delay().

        Returns:
            DataFrame with columns: applicable_delay, unit
            Index: indicator or MultiIndex(entity, indicator)

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
            >>> # Panel data example
            >>> delays = {('France', 'GDP'): 45.0, ('Germany', 'GDP'): 38.0}
            >>> transformer = PublicationDelayTransformer(delays=delays, panel_cols=['country'])
            >>> transformer.fit(X)
            >>> df = transformer.get_delays_as_dataframe()
            >>> print(df.index.names)
            ['entity', 'indicator']
        """
        if self.delays_dict_ is None:
            raise RuntimeError("Transformer has not been fitted yet")

        return delays_dict_to_dataframe(
            self.delays_dict_,
            unit=getattr(self, '_delays_unit', 'D')
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


# Alias pour compatibilité arrière
ReleaseDelayTransformer = PublicationDelayTransformer