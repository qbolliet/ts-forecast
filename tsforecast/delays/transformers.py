"""Sklearn-compatible transformers for applying publication delays.

This module provides a modular architecture with:
- ShiftTransformer: Pure helper to shift data by N periods
- MaskTransformer: Pure helper to mask N observations per period
- PublicationDelayTransformer: Intelligent orchestrator that handles inference, frequency detection, and panel wrapping
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
import math
from typing import Any, Callable, Dict, Optional, Union, List, Literal
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Importation des modules du package
from tsforecast.utils.frequency import normalize_frequency, to_pandas_freq
from tsforecast.utils.time import resolve_date, get_period_boundaries
from tsforecast.utils.duration import convert_duration, normalize_duration
from ..frequency.detector import detect_frequency
from ..panel import PanelwiseTransformer
from .auxiliary_transformers import ShiftTransformer, MaskTransformer


# Classe d'application des délais de publication
class PublicationDelayTransformer(BaseEstimator, TransformerMixin):
    """Intelligent orchestrator for applying publication delays to time series/panel data.

    This transformer handles:
    - Parameter inference from delays DataFrame
    - Frequency detection per column
    - Period-based calculations (not day-based)
    - Automatic panel wrapping with PanelwiseTransformer
    - Warning generation for all-NaN columns

    Parameters:
        delays: Delays specification (Dict or DataFrame)
        strategy: Transformation strategy ('shift' or 'mask')
        delay_unit: Unit of delay ('D', 's', 'h', etc.). If None, inferred from DataFrame
        reference_point: Reference point ('start' or 'end'). If None, inferred from DataFrame
        target_frequency: Target frequency for delay calculation. If None, uses column frequency
        prediction_date: Date of prediction (required for 'mask' strategy)
        time_col: Name of time column (default: 'date')
        panel_cols: Panel column names (None for non-panel data)
        handle_missing_delays: Strategy for missing delays ('ignore', 'warn', 'error')
        default_delay: Default delay value if missing

    Attributes:
        column_transformers_: Dict mapping column names to helper transformers
        inferred_params_: Dict of parameters inferred from delays DataFrame
        detected_frequencies_: Dict of detected frequencies per column

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>>
        >>> # Create delays DataFrame with metadata
        >>> delays_df = pd.DataFrame({
        ...     'variable': ['GDP', 'inflation'],
        ...     'delay': [45.0, 30.0],
        ...     'unit': ['D', 'D'],
        ...     'reference_point': ['end', 'end'],
        ...     'target_frequency': ['M', 'M']
        ... })
        >>>
        >>> # Create transformer (parameters inferred from DataFrame)
        >>> transformer = PublicationDelayTransformer(
        ...     delays=delays_df,
        ...     strategy='shift',
        ...     prediction_date=datetime(2024, 12, 15)
        ... )
        >>>
        >>> # Apply transformation
        >>> X_shifted = transformer.fit_transform(X)
        >>>
        >>> # Reverse transformation
        >>> X_original = transformer.inverse_transform(X_shifted)
    """

    # Initialisation
    def __init__(
        self,
        delays: Union[Dict[str, float], pd.DataFrame],
        prediction_date: Union[str, datetime] = 'today',
        strategy: Union[Literal['shift', 'mask'], Dict[str, Literal['shift', 'mask']]] = 'shift',
        target_frequency: Optional[Union[str, Dict[str, str]]] = None,
        delay_unit: Optional[Union[str, Dict[str, str]]] = None,
        reference_point: Optional[Union[Literal['start', 'end'], Dict[str, Literal['start', 'end']]]] = None,
        time_col: str = 'date',
        panel_cols: Optional[List[str]] = None,
        handle_missing_delays: Literal['ignore', 'warn', 'error'] = 'warn',
        default_values: Optional[Dict[str, Union[int, float, str]]] = None
    ):
        """Initialize PublicationDelayTransformer.

        Args:
            delays: Dict mapping variable names to delays, or DataFrame with delays
            prediction_date: Prediction date
            strategy: 'shift' or 'mask' or dictionnary mapping variables names to strategies. Default delay is ignored when strategies are dictionnaries
            target_frequency: Target frequency for mask strategy
            delay_unit: Unit of delay (inferred from DataFrame if None)
            reference_point: Delay reference point, 'start' or 'end' (inferred from DataFrame if None)
            time_col: Time column name
            panel_cols: Panel column names
            handle_missing_delays: 'ignore', 'ignore', or 'error'
            default_values: Default delay if missing
        """
        # Validation des paramètres
        # Paramètre de stratégie
        if isinstance(strategy, str) :
            if strategy not in ['shift', 'mask']:
                raise ValueError(f"strategy must be 'shift' or 'mask', got '{strategy}'")
        elif isinstance(strategy, dict):
            # Parcours des valeurs :
            for k, v in strategy.items():
                if v not in ['shift', 'mask']:
                    raise ValueError(f"strategy must be 'shift' or 'mask', for variable '{k}' got '{v}'")
        else:
            raise TypeError(f"'strategy' should be a string of a dictionnary, got a {type(strategy)}")
        
        # Paramètre de point de référence
        if reference_point is not None and reference_point not in ['start', 'end']:
            raise ValueError(f"reference_point must be 'start' or 'end', got '{reference_point}'")
        
        # Gestion des délais manquants
        if handle_missing_delays not in ['ignore', 'warn', 'error']:
            raise ValueError(f"'handle_missing_delays' must be 'ignore', 'warn', or 'error', got '{handle_missing_delays}'")

        # Paramètre de délai par défaut
        if default_values is not None :
            # Clés attendues
            expected_keys = ['delay', 'unit', 'reference_point'] if (strategy != 'mask') else ['delay', 'unit', 'reference_point', 'target_frequency'] 
            # Clés manquantes
            missing_default_delay_keys = set(expected_keys) - set(default_values.keys())
            if len(missing_default_delay_keys) > 0:
                raise ValueError(f"Expected a 'default_values' dictionnary with the keys : {expected_keys}, the following keys are missing{list(missing_default_delay_keys)}")

        # Stockage des paramètres
        self.delays = delays
        self.prediction_date = prediction_date
        self.strategy = strategy
        self.target_frequency = target_frequency
        self.delay_unit = delay_unit
        self.reference_point = reference_point
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.handle_missing_delays = handle_missing_delays
        self.default_values = default_values

        # Warnings
        if isinstance(strategy, dict) and (default_values is not None):
            warnings.warn("'default_values' is ignored when the strategy is specified as a dictionnary")
        if (strategy == 'shift') and (target_frequency is not None):
            warnings.warn("'target_frequency' is ignored when a shifting strategy is applied")

    # Méthode d'entraînement
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Fit transformer by inferring parameters and preparing helpers.

        Args:
            X: Time series or panel data
            y: Ignored

        Returns:
            self
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(self.prediction_date)

        # Inférence des paramètres depuis delays DataFrame si nécessaire
        self.inferred_params_ = self._infer_parameters_from_delays()

        # Détermination des valeurs des paramètres (explicites > inférées > défaut)
        # Fréquence cible
        # Initialisation avec les paramètres inférés
        target_frequency_dict = self.inferred_params_['target_frequency']
        # Mise à jour avec les paramètres spécifiés
        if isinstance(self.target_frequency, dict):
            target_frequency_dict.update(self.target_frequency)
        elif isinstance(self.target_frequency, str):
            target_frequency_dict.update({c : self.target_frequency for c in X.columns})
        # Ajout de la valeur par défaut pour les colonnes restantes
        if self.default_values is not None :
            # Détection des variables qui n'ont pas de target frequency
            missing_target_frequency = set(X.columns) - set(target_frequency_dict.keys())
            # Si les stratégies sont fournies sous forme de dictionnaire, on vérifie que les variables pour lesquelles la fréquence est manquante sont des 'mask'
            if isinstance(self.strategy, dict) and ('target_frequency' in self.default_values.keys()):
                missing_target_frequency_strategy = missing_target_frequency - set([k for k,v in self.strategy.items() if v == 'shift'])
                if len(missing_target_frequency_strategy) > 0: # Ce if est possiblement inutile
                    # Ajout de la valeur par défaut
                    for col in missing_target_frequency_strategy :
                        target_frequency_dict[col] = self.default_values['target_frequency']
                        # Warning
                        warnings.warn(f"Imputed default target frequency value '{self.default_values['target_frequency']}' for column '{col}'")
            # Cas où toutes les variables doivent être masquées
            elif (self.strategy=="mask") and ('target_frequency' in self.default_values.keys()) and (len(missing_target_frequency) > 0):
                # Ajout de la valeur par défaut
                for col in missing_target_frequency :
                    target_frequency_dict[col] = self.default_values['target_frequency']
                    # Warning
                    warnings.warn(f"Imputed default target frequency value '{self.default_values['target_frequency']}' for column '{col}'")
            # Cas où il y aurait des variables à imputer mais qu'un fréquence par défaut n'est pas spécifiée
            elif ('target_frequency' not in self.default_values.keys()) and (len(missing_target_frequency)>0):
                # Distinction suivant la stratégie
                if isinstance(self.strategy, dict):
                    missing_target_frequency_strategy = missing_target_frequency - set([k for k,v in self.strategy.items() if v == 'shift'])
                    if len(missing_target_frequency_strategy) > 0:
                        # Warning
                        warnings.warn(f"Could not impute a default 'target_frequency' for columns {missing_target_frequency_strategy} because it is not specified in the 'default_values' dictionnary")
                elif self.strategy == "mask" :
                    # Warning
                    warnings.warn(f"Could not impute a default 'target_frequency' for columns : {missing_target_frequency} because it is not specified in the 'default_values' dictionnary")
        # Tous les autres cas ("shift"), absence de valeur par défaut, absence de variable pour laquelle la "target_frequency" n'est pas spécifiée sont normaux et ne nécessitent ni warning ni imputation
        
        
        # Unité des délais
        # Initialisation avec les paramètres inférés
        delay_unit_dict = self.inferred_params_['delay_unit']
        # Mise à jour avec les paramètres spécifiés
        if isinstance(self.delay_unit, dict):
            delay_unit_dict.update(self.delay_unit)
        elif isinstance(self.delay_unit, str):
            delay_unit_dict.update({c : self.delay_unit for c in X.columns})
        # Ajout de la valeur par défaut pour les colonnes restantes
        if self.default_values is not None :
            # Détection des variables qui n'ont pas d'unité
            missing_delay_unit = set(X.columns) - set(delay_unit_dict.keys())
            # Cas où le répertoire des stratégies est un dictionnaire
            if isinstance(self.strategy, dict) and ('delay_unit' in self.default_values.keys()):
                missing_delay_unit_strategy = missing_delay_unit - set(self.strategy.keys())
                if len(missing_delay_unit_strategy) > 0: # Ce if est possiblement inutile
                    # Ajout de la valeur par défaut
                    for col in missing_delay_unit_strategy :
                        delay_unit_dict[col] = self.default_values['delay_unit']
                        # Warning
                        warnings.warn(f"Imputed default delay unit value '{self.default_values['delay_unit']}' for column '{col}'")
            # Cas où la valeur par défaut doit être associée à toutes les colonnes non référencées 
            elif ('delay_unit' in self.default_values.keys()) and (len(missing_delay_unit) > 0):
                # Ajout de la valeur par défaut
                for col in missing_delay_unit :
                    delay_unit_dict[col] = self.default_values['delay_unit']
                    # Warning
                    warnings.warn(f"Imputed default delay unit value '{self.default_values['delay_unit']}' for column '{col}'")
            # Cas où il y aurait des variables à imputer mais qu'un fréquence par défaut n'est pas spécifiée
            elif ('delay_unit' not in self.default_values.keys()) and (len(missing_delay_unit)>0):
                warnings.warn(f"Could not impute a default 'delay_unit' for columns {missing_delay_unit} because it is not specified in the 'default_values' dictionnary")
        # Tous les autres cas, absence de valeur par défaut, absence de variable pour laquelle le "delay_unit" n'est pas spécifiée sont normaux et ne nécessitent ni warning ni imputation
        
        # Point de référence
        # Initialisation avec les paramètres inférés
        reference_point_dict = self.inferred_params_['reference_point']
        # Mise à jour avec les paramètres spécifiés
        if isinstance(self.reference_point, dict):
            reference_point_dict.update(self.reference_point)
        elif isinstance(self.reference_point, str):
            reference_point_dict.update({c : self.reference_point for c in X.columns})
        # Ajout de la valeur par défaut pour les colonnes restantes
        if self.default_values is not None :
            # Détection des variables qui n'ont pas d'unité
            missing_reference_point = set(X.columns) - set(reference_point_dict.keys())
            # Cas où le répertoire des stratégies est un dictionnaire
            if isinstance(self.strategy, dict) and ('reference_point' in self.default_values.keys()):
                missing_reference_point_strategy = missing_reference_point - set(self.strategy.keys())
                if len(missing_reference_point_strategy) > 0: # Ce if est possiblement inutile
                    # Ajout de la valeur par défaut
                    for col in missing_reference_point_strategy :
                        reference_point_dict[col] = self.default_values['reference_point']
                        # Warning
                        warnings.warn(f"Imputed default delay unit value '{self.default_values['reference_point']}' for column '{col}'")
            # Cas où la valeur par défaut doit être associée à toutes les colonnes non référencées 
            elif ('reference_point' in self.default_values.keys()) and (len(missing_reference_point) > 0):
                # Ajout de la valeur par défaut
                for col in missing_reference_point :
                    reference_point_dict[col] = self.default_values['reference_point']
                    # Warning
                    warnings.warn(f"Imputed default delay unit value '{self.default_values['reference_point']}' for column '{col}'")
            # Cas où il y aurait des variables à imputer mais qu'un fréquence par défaut n'est pas spécifiée
            elif ('reference_point' not in self.default_values.keys()) and (len(missing_reference_point)>0):
                warnings.warn(f"Could not impute a default 'reference_point' for columns {missing_reference_point} because it is not specified in the 'default_values' dictionnary")
        # Tous les autres cas, absence de valeur par défaut, absence de variable pour laquelle le "reference_point" n'est pas spécifiée sont normaux et ne nécessitent ni warning ni imputation
        
        # Conversion des delays en dictionnaire si DataFrame
        if isinstance(self.delays, pd.DataFrame):
            delays_dict = dict(zip(self.delays['variable'], self.delays['delay']))
        else:
            delays_dict = self.delays

        # Enumération des variables auxquelles appliquer une stratégie de 'shift' et de 'mask'
        if isinstance(self.strategy, str):
            # Distinction suivant la stratégie à appliquer
            if self.strategy == 'shift' :
                shift_columns = np.intersect1d(X.columns.tolist(), list(delays_dict.keys())).tolist() if self.default_delay is None else X.columns.tolist()
                mask_columns = []
            else : # équivalent à self.strategy == 'mask'
                mask_columns = np.intersect1d(X.columns.tolist(), list(delays_dict.keys())).tolist() if self.default_delay is None else X.columns.tolist()
                shift_columns = []

        else : # équivalent à isinstance(self.strategy, dict)
            shift_columns = np.intersect1d(X.columns.tolist(), [k for k,v in self.strategy if v == 'shift']).tolist()
            mask_columns = np.intersect1d(X.columns.tolist(), [k for k,v in self.strategy if v == 'mask']).tolist()

        # Détection des fréquences par colonne
        self.detected_frequencies_ = detect_frequency(data=X, time_col=self.time_col, panel_cols=None, literal = False, check_consistency= True)

        # Calcul du nombre de périodes à shifter pour chaque variable
        # Initialisation du dictionnaire résultat
        self.shift_params = {}
        # Parcours des variables
        for col in shift_columns:
            # Normalisation de l'unité des délais
            delay_unit = normalize_duration(delay_unit_dict[col])

            # Calcul des bornes de la période associée à la date de prédiction
            period_start, period_end = get_period_boundaries(self.prediction_date_, self.detected_frequencies_[col])
            # Calcul du temps écoulé, dans l'unité du délai, entre la date de prédiction et le début de la période
            elapsed_duration = convert_duration(
                value=(self.prediction_date_ - period_start).to_seconds(),
                from_duration='s',
                to_duration=delay_unit,
                rounding=None
            )
            
            # Conversion de la durée de la période dans l'unité du délai
            period_duration = convert_duration(
                value=1,
                from_duration=self.detected_frequencies_[col],
                to_duration=delay_unit,
                rounding=None
            )

            # Si le point de référence de calcul du délai est la fin, on lui ajoute la durée de la période
            if reference_point_dict[col] == 'end':
                elapsed_duration += period_duration

            # On calcule l'arrondi à l'unité supérieure de la différence entre le délai et la date de prédiction, divisée par la longueur de la période associée à la fréquence de la série
            n_periods = math.ceil((delays_dict[col] - elapsed_duration) / period_duration)
             
            # Ajout au dictionnaire résultat
            self.shift_params[col] = {'n_periods': n_periods, 'frequency': self.detected_frequencies_[col]}


        # Calcul du nombre d'observations à masquer pour chaque variable
        # Initialisation du dictionnaire résultat
        self.mask_params = {}
        # Parcours des variables
        for col in mask_columns:
            # Normalisation de l'unité des délais
            delay_unit = normalize_duration(delay_unit_dict[col])
            # Normalisation de la fréquence cible
            target_frequency = normalize_frequency(target_frequency_dict[col])

            # Calcul des bornes de la période associée à la date de prédiction
            period_start, period_end = get_period_boundaries(self.prediction_date_, self.detected_frequencies_[col])
            # Calcul du temps écoulé, dans l'unité du délai, entre la date de prédiction et le début de la période
            elapsed_duration = convert_duration(
                value=(self.prediction_date_ - period_start).to_seconds(),
                from_duration='s',
                to_duration=delay_unit,
                rounding=None
            )
            
            # Conversion de la durée de la période dans l'unité du délai
            period_duration = convert_duration(
                value=1,
                from_duration=self.detected_frequencies_[col],
                to_duration=delay_unit,
                rounding=None
            )

            # Si le point de référence de calcul du délai est la fin, on lui ajoute la durée de la période
            if reference_point_dict[col] == 'end':
                elapsed_duration += period_duration

            # On calcule l'arrondi à l'unité supérieure de la différence entre le délai et la date de prédiction, divisée par la longueur de la période associée à la fréquence de la série
            n_periods = math.ceil((delays_dict[col] - elapsed_duration) / period_duration)

            # Calcul du nombre d'observations à la fréquence de la série il y a dans la période à la fréquence cible
            target_period_duration = convert_duration(
                value=1,
                from_duration=target_frequency,
                to_duration=self.detected_frequencies_[col],
                rounding=None
            )
            # Vérification que le nombre d'observations à masquer est bien strictement inférieur au nombre d'observations dans la période à la 'target_frequency'
            if (math.floor(target_period_duration) > n_periods):
                # Ajout au dictionnaire des variables à mask
                self.mask_params[col] = {'n_obs': n_periods, 'mask_frequency': target_frequency, "how": "last"}
            # Sinon, on met un warning et on ajoute au dictionnaire de shifts
            else :
                # Warning
                warnings.warn(f"Could not mask the column '{col}' because it would have created a series of Nan. Moved it to the shifted columns")
                # Ajout du dictionnaire des variables à shift
                self.shift_params[col] = {'n_periods': n_periods, 'frequency': self.detected_frequencies_[col]}

        return self

    # Méthode de transformation des données
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply publication delays to data.

        Args:
            X: Time series or panel data

        Returns:
            Transformed data with publication delays applied
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self)

        # Détection de la structure de panel
        # Des transformations similaires sont appliquées à tous les individus du panel. (L'AJOUTER EN DOCSTRING)
        

        if self.is_series_:
            # Transform series
            column_name = X.name or 'series'
            if column_name in self.column_transformers_:
                result = self.column_transformers_[column_name].transform(X)
                # Check for all-NaN
                if result.isna().all():
                    warnings.warn(
                        f"Column '{column_name}' became all-NaN after applying {self.strategy} strategy"
                    )
                return result
            else:
                return X.copy()
        else:
            # Transform DataFrame column by column
            X_result = X.copy()
            for column_name, transformer in self.column_transformers_.items():
                if column_name in X_result.columns:
                    X_result[column_name] = transformer.transform(X_result[column_name])
                    # Check for all-NaN
                    if X_result[column_name].isna().all():
                        warnings.warn(
                            f"Column '{column_name}' became all-NaN after applying {self.strategy} strategy"
                        )
            return X_result

    # Méthode de transformation inverse des données
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse publication delay transformation.

        Args:
            X: Transformed data

        Returns:
            Data with delays reversed
        """
        if self.is_series_:
            # Inverse transform series
            column_name = X.name or 'series'
            if column_name in self.column_transformers_:
                return self.column_transformers_[column_name].inverse_transform(X)
            else:
                return X.copy()
        else:
            # Inverse transform DataFrame column by column
            X_result = X.copy()
            for column_name, transformer in self.column_transformers_.items():
                if column_name in X_result.columns:
                    X_result[column_name] = transformer.inverse_transform(X_result[column_name])
            return X_result

    # Méthode auxiliaire d'inférence des paramètres d'unité du délai, de point de référence et de fréquence cible
    def _infer_parameters_from_delays(self) -> Dict[str, any]:
        """Infer delay_unit, reference_point, target_frequency from delays DataFrame.

        Returns:
            Dict of inferred parameters
        """
        # Initialisation du dictionnaire résultat
        inferred = {}

        # Extraction des éléments du jeu de données
        # Extrait à chaque fois la première valeur en faisant l'hypothèse qu'elle est constante
        if isinstance(self.delays, pd.DataFrame):
            # Inférence de 'delay_unit' à partir de la colonne 'unit'
            if 'unit' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et l'unité
                df_unit = self.delays[['variable', 'unit']].drop_duplicates(subset=['variable'])
                inferred['delay_unit'] = dict(
                    zip(df_unit['variable'], df_unit['unit'])
                )

            # Inférence de 'reference_point' à partir de la colonne 'reference_point'
            if 'reference_point' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et lde point de référence
                df_reference_point = self.delays[['variable', 'reference_point']].drop_duplicates(subset=['variable'])
                inferred['reference_point'] = dict(
                    zip(df_reference_point['variable'], df_reference_point['reference_point'])
                )

            # Inférence de 'target_frequency' à partir de la colonne 'target_frequency'
            if 'target_frequency' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et la fréquence
                df_frequency = self.delays[['variable', 'target_frequency']].drop_duplicates(subset=['variable'])
                inferred['target_frequency'] = dict(
                    zip(df_frequency['variable'], df_frequency['target_frequency'])
                )

        return inferred

    # Méthode auxiliaire d'entrainement du transformer
    def _fit_column_transformer(
        self,
        column_name: str,
        series: pd.Series,
        delays_dict: Dict[str, float],
        delay_unit: str,
        reference_point: str
    ):
        """Fit helper transformer for a single column.

        Args:
            column_name: Name of the column
            series: Series data for the column
            delays_dict: Dictionary of delays
            delay_unit: Unit of delay
            reference_point: Reference point
        """
        # Obtention du délai pour cette colonne
        if column_name in delays_dict:
            applicable_delay = delays_dict[column_name]
        elif self.default_delay is not None:
            applicable_delay = self.default_delay
        else:
            if self.handle_missing_delays == 'error':
                raise ValueError(f"No delay found for column '{column_name}'")
            elif self.handle_missing_delays == 'warn':
                warnings.warn(f"No delay found for column '{column_name}', skipping transformation")
            return  # Skip this column

        # Obtention de la fréquence détectée
        column_frequency = self.detected_frequencies_.get(column_name)
        if column_frequency is None:
            warnings.warn(f"Could not detect frequency for column '{column_name}', skipping transformation")
            return

        # Détermination de la target_frequency
        target_freq = None
        if self.target_frequency is not None:
            target_freq = self.target_frequency
        elif 'target_frequencies' in self.inferred_params_ and column_name in self.inferred_params_['target_frequencies']:
            target_freq = self.inferred_params_['target_frequencies'][column_name]
        # Si target_freq reste None, calculate_n_periods_delay utilisera column_frequency

        # Calcul du nombre de périodes
        from .period_utils import calculate_n_periods_delay

        # Utilisation d'une observation typique pour le calcul
        observation_date = series.index[len(series) // 2]

        n_periods = calculate_n_periods_delay(
            applicable_delay=applicable_delay,
            delay_unit=delay_unit,
            prediction_date=self.prediction_date_,
            reference_point=reference_point,
            observation_date=observation_date,
            column_frequency=column_frequency,
            target_frequency=target_freq
        )

        # Création du helper transformer approprié
        if self.strategy == 'shift':
            helper = ShiftTransformer(
                n_periods=n_periods,
                frequency=target_freq or column_frequency
            )
        else:  # mask
            helper = MaskTransformer(
                n_obs=abs(n_periods),  # MaskTransformer expects positive integer
                mask_frequency=target_freq or column_frequency,
                prediction_date=self.prediction_date_
            )

        # Stockage du transformer
        self.column_transformers_[column_name] = helper


# Fonction de création d'une factory de PublicationDelayTransformer pour l'utilisation sur des données de panel
def create_delay_transformer_factory(
    df_delays: pd.DataFrame,
    strategy: Union[
        Literal['shift', 'mask'],
        Dict[Union[str, tuple], Literal['shift', 'mask']],
        Callable[[tuple], Literal['shift', 'mask']]
    ] = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    delay_col: str = 'applicable_delay',
    unit_col: str = 'unit',
    reference_point_col: str = 'target_reference_point',
    target_frequency_col: str = 'target_frequency',
    default_transformer_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[tuple], PublicationDelayTransformer]:
    """Create a transformer factory from a publication delays DataFrame.

    This function generates a callable factory that creates entity-specific
    PublicationDelayTransformer instances, suitable for use with
    PanelwiseTransformer. The factory extracts delay parameters for each
    entity from the provided DataFrame.

    Args:
        df_delays: DataFrame from calculate_applicable_delay() with
            aggregate_by_panel=True, expected to have a MultiIndex with
            panel entity as the first levels and variable as the last level, and columns for delay values
            and metadata.
        strategy: Delay application strategy. Can be:
            - str: 'shift' or 'mask' applied to all entities
            - Dict[tuple, str]: Mapping of entity keys to strategies
            - Callable[[tuple], str]: Function returning strategy for entity
        prediction_date: Date of prediction for delay calculations.
            Passed to PublicationDelayTransformer.
        panel_level: Index level name or position for panel entities.
            Defaults to 0 (first level).
        variable_level: Index level name or position for variables.
            Defaults to -1 (last level).
        delay_col: Column name for delay values. Defaults to 'applicable_delay'.
        unit_col: Column name for delay units. Defaults to 'unit'.
        reference_point_col: Column name for reference point.
            Defaults to 'target_reference_point'.
        target_frequency_col: Column name for target frequency.
            Defaults to 'target_frequency'.
        default_transformer_kwargs: Additional kwargs passed to all
            PublicationDelayTransformer instances (e.g., time_col, panel_cols).

    Returns:
        Callable that takes an entity_key (tuple) and returns a configured
        transformer instance for that entity.

    Raises:
        ValueError: If required columns are missing from df_delays.
        KeyError: If entity not found in df_delays (at factory call time).

    Examples:
        Basic usage with uniform strategy:

        >>> # Calculate delays with panel aggregation
        >>> delays = calculate_applicable_delay(
        ...     publication_delays=raw_delays,
        ...     target_reference_point='end',
        ...     target_frequency='M',
        ...     aggregate_by_panel=True
        ... )
        >>>
        >>> # Create factory
        >>> factory = create_delay_transformer_factory(
        ...     df_delays=delays,
        ...     strategy='shift',
        ...     prediction_date='2024-12-15'
        ... )
        >>>
        >>> # Use with PanelwiseTransformer
        >>> panelwise = PanelwiseTransformer(
        ...     transformer=factory,
        ...     panel_cols=['country'],
        ...     time_col='date'
        ... )
        >>> X_transformed = panelwise.fit_transform(X)

        Entity-specific strategies via dict:

        >>> factory = create_delay_transformer_factory(
        ...     df_delays=delays,
        ...     strategy={
        ...         ('FR',): 'shift',
        ...         ('DE',): 'mask',
        ...         ('IT',): 'shift'
        ...     },
        ...     prediction_date='2024-12-15'
        ... )

        Entity-specific strategies via callable:

        >>> def strategy_selector(entity_key):
        ...     # Use mask for entities with short delays
        ...     if entity_key in high_frequency_entities:
        ...         return 'mask'
        ...     return 'shift'
        >>>
        >>> factory = create_delay_transformer_factory(
        ...     df_delays=delays,
        ...     strategy=strategy_selector,
        ...     prediction_date='2024-12-15'
        ... )

    Notes:
        - The factory caches parsed entity configurations for efficiency
        - Missing entities raise KeyError with helpful error message
        - All transformers share the same prediction_date
        - Strategy can vary per entity while other params come from DataFrame
    """
    # Validation des colonnes requises
    required_cols = [delay_col, unit_col, reference_point_col, target_frequency_col]
    missing_cols = [col for col in required_cols if col not in df_delays.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in df_delays: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )

    # Validation de l'index
    if not isinstance(df_delays.index, pd.MultiIndex):
        raise ValueError(
            "df_delays must have a MultiIndex (panel_entity, variable). "
            "Use calculate_applicable_delay with aggregate_by_panel=True."
        )

    # Calcul des paramètres pour chaque entité
    entity_params = _build_entity_params(
        df_delays=df_delays,
        delay_col=delay_col,
        unit_col=unit_col,
        reference_point_col=reference_point_col,
        target_frequency_col=target_frequency_col
    )


    # Préparation des kwargs par défaut
    base_kwargs = default_transformer_kwargs.copy() if default_transformer_kwargs else {}
    base_kwargs['prediction_date'] = prediction_date

    # Création de la factory
    def transformer_factory(entity_key: tuple) -> BaseEstimator:
        """Create a configured transformer for the specified entity.

        Args:
            entity_key: Entity identifier as tuple.

        Returns:
            Configured transformer instance.

        Raises:
            KeyError: If entity not found in delays configuration.
        """
        # Normalisation de la clé
        if not isinstance(entity_key, tuple):
            entity_key = (entity_key,)

        # Vérification de l'existence de l'entité
        if entity_key not in entity_params:
            available = list(entity_params.keys())[:10]
            more = f"... and {len(entity_params) - 10} more" if len(entity_params) > 10 else ""
            raise KeyError(
                f"Entity {entity_key} not found in delays configuration. "
                f"Available entities: {available}{more}"
            )

        # Récupération de la configuration de l'entité
        params = entity_params[entity_key]

        # Détermination de la stratégie pour cette entité
        entity_strategy = _resolve_strategy(strategy, entity_key)

        # Construction des kwargs du transformer
        transformer_kwargs = {
            **base_kwargs,
            'delays': params['delays'],
            'delay_unit': params['delay_unit'],
            'reference_point': params['reference_point'],
            'target_frequency': params['target_frequency'],
            'strategy': entity_strategy
        }

        # Création et retour du transformer
        return PublicationDelayTransformer(**transformer_kwargs)

    return transformer_factory


# Fonction auxiliaire de construction des dictionnaires de paramètres pour chaque entité
def _build_entity_params(
    df_delays: pd.DataFrame,
    delay_col: str,
    unit_col: str,
    reference_point_col: str,
    target_frequency_col: str
) -> Dict[tuple, Dict[str, Any]]:
    """Build parameters dictionaries for each entity.

    Args:
        df_delays: Source DataFrame with delays.
        panel_level_name: Name of panel entity level.
        variable_level_name: Name of variable level.
        delay_col: Column name for delays.
        unit_col: Column name for units.
        reference_point_col: Column name for reference points.
        target_frequency_col: Column name for frequencies.

    Returns:
        Dict mapping entity keys to configuration dicts.
    """
    # Initialisation du dictionnaire de paramètres associés à l'entité
    entity_params = {}

    # Groupement par entité panel
    for entity_key, group in df_delays.groupby(level=list(range(df_delays.index.nlevels - 1))):
        # Normalisation de la clé en tuple
        entity_key = normalize_entity_key(entity_key)

        # Construction du dictionnaire de délais (variable -> delay)
        delays_dict = dict(zip(
            group.index.get_level_values(df_delays.index.nlevels - 1),
            group[delay_col]
        ))

        # Construction du dictionnaire de paramètres
        entity_params[entity_key] = {
            'delays': delays_dict,
            'delay_unit': _extract_param_by_variable(group, unit_col),
            'reference_point': _extract_param_by_variable(group, reference_point_col),
            'target_frequency': _extract_param_by_variable(group, target_frequency_col)
        }

    return entity_params

# Fonction auxiliaire d'extraction de la variable
def _extract_param_by_variable(
    group: pd.DataFrame,
    column: str
) -> Union[str, Dict[str, str]]:
    """Extract parameter, returning dict if varies by variable.

    Args:
        group: DataFrame group for one entity.
        col: Column to extract.
        variable_level_name: Name of variable index level.

    Returns:
        Single value if constant, or dict mapping variable to value.
    """
    # Vérification de l'unicité de la valeur
    unique_values = group[column].unique()

    if len(unique_values) == 1:
        # Valeur constante pour toutes les variables
        return unique_values[0]
    else:
        # Valeur variable selon les variables -> retourne un dictionnaire
        return dict(zip(
            group.index.get_level_values(group.index.nlevels - 1),
            group[column]
        ))

# A REVOIR
def _resolve_strategy(
    strategy: Union[str, Dict[tuple, str], Callable[[tuple], str]],
    entity_key: tuple
) -> str:
    """Resolve strategy for a specific entity.

    Args:
        strategy: Strategy specification (str, dict, or callable).
        entity_key: Entity identifier.

    Returns:
        Strategy string ('shift' or 'mask') for the entity.

    Raises:
        ValueError: If strategy is invalid.
        KeyError: If entity not found in strategy dict.
    """
    if isinstance(strategy, str):
        # Stratégie globale
        if strategy not in ('shift', 'mask'):
            raise ValueError(f"Invalid strategy: '{strategy}'. Must be 'shift' or 'mask'.")
        return strategy

    elif isinstance(strategy, dict):
        # Dictionnaire de stratégies par entité
        if entity_key not in strategy:
            # Tentative avec clé non-tuple si entité simple
            if len(entity_key) == 1 and entity_key[0] in strategy:
                return strategy[entity_key[0]]
            raise KeyError(
                f"No strategy defined for entity {entity_key}. "
                f"Available entities in strategy dict: {list(strategy.keys())}"
            )
        return strategy[entity_key]

    elif callable(strategy):
        # Callable qui retourne la stratégie
        result = strategy(entity_key)
        if result not in ('shift', 'mask'):
            raise ValueError(
                f"Strategy callable returned invalid value '{result}' for entity {entity_key}. "
                "Must return 'shift' or 'mask'."
            )
        return result

    else:
        raise TypeError(
            f"strategy must be str, dict, or callable, got {type(strategy).__name__}"
        )


def prepare_entity_kwargs_from_delays(
    df_delays: pd.DataFrame,
    strategy: Union[
        Literal['shift', 'mask'],
        Dict[tuple, Literal['shift', 'mask']]
    ] = 'shift',
    panel_level: Union[str, int] = 0,
    variable_level: Union[str, int] = -1,
    delay_col: str = 'applicable_delay',
    unit_col: str = 'unit',
    reference_point_col: str = 'target_reference_point',
    target_frequency_col: str = 'target_frequency'
) -> Dict[tuple, Dict[str, Any]]:
    """Prepare entity_kwargs dict from a publication delays DataFrame.

    This is an alternative to create_delay_transformer_factory() for use
    with PanelwiseTransformer's entity_kwargs parameter instead of the
    factory pattern.

    Args:
        df_delays: DataFrame from calculate_applicable_delay() with
            aggregate_by_panel=True.
        strategy: Delay strategy ('shift' or 'mask'), or dict mapping
            entity keys to strategies.
        panel_level: Index level for panel entities.
        variable_level: Index level for variables.
        delay_col: Column name for delays.
        unit_col: Column name for units.
        reference_point_col: Column name for reference points.
        target_frequency_col: Column name for frequencies.

    Returns:
        Dict mapping entity keys to kwargs dicts suitable for set_params().

    Examples:
        >>> entity_kwargs = prepare_entity_kwargs_from_delays(
        ...     df_delays=calculated_delays,
        ...     strategy={'FR': 'shift', 'DE': 'mask'}
        ... )
        >>>
        >>> panelwise = PanelwiseTransformer(
        ...     transformer=PublicationDelayTransformer(
        ...         strategy='shift',  # Default, overridden by entity_kwargs
        ...         prediction_date='2024-12-15',
        ...         delays={}
        ...     ),
        ...     entity_kwargs=entity_kwargs,
        ...     panel_cols=['country']
        ... )

    Notes:
        - This approach is simpler but less flexible than the factory pattern
        - Requires base transformer to support all entity-specific params via set_params()
        - Does not support callable strategy selectors (use factory for that)
    """
    # Validation des colonnes requises
    required_cols = [delay_col, unit_col, reference_point_col, target_frequency_col]
    missing_cols = [col for col in required_cols if col not in df_delays.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in df_delays: {missing_cols}"
        )

    # Résolution des noms de niveaux
    panel_level_name = _resolve_index_level(df_delays.index, panel_level)
    variable_level_name = _resolve_index_level(df_delays.index, variable_level)

    # Construction des configs
    entity_configs = _build_entity_configs(
        df_delays=df_delays,
        panel_level_name=panel_level_name,
        variable_level_name=variable_level_name,
        delay_col=delay_col,
        unit_col=unit_col,
        reference_point_col=reference_point_col,
        target_frequency_col=target_frequency_col
    )

    # Conversion en entity_kwargs format
    entity_kwargs = {}
    for entity_key, config in entity_configs.items():
        # Résolution de la stratégie
        if isinstance(strategy, dict):
            # Recherche dans le dict avec gestion des clés non-tuple
            if entity_key in strategy:
                entity_strategy = strategy[entity_key]
            elif len(entity_key) == 1 and entity_key[0] in strategy:
                entity_strategy = strategy[entity_key[0]]
            else:
                raise KeyError(f"No strategy defined for entity {entity_key}")
        else:
            entity_strategy = strategy

        # Construction des kwargs
        entity_kwargs[entity_key] = {
            'delays': config['delays'],
            'delay_unit': config['delay_unit'],
            'reference_point': config['reference_point'],
            'target_frequency': config['target_frequency'],
            'strategy': entity_strategy
        }

    return entity_kwargs