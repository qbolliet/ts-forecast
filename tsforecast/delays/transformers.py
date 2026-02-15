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
from typing import Any, Callable, Dict, Optional, Union, List, Literal, Tuple
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Importation des modules du package
from tsforecast.utils.frequency import (
    normalize_frequency,
    is_higher_frequency
)
from tsforecast.frequency import (
    detect_index_frequency,
    detect_and_parse_index_frequency,
    build_frequency_string
)
from tsforecast.utils.time import resolve_date, get_period_start, get_period_boundaries
from tsforecast.utils.duration import convert_duration, normalize_duration, DurationConverter
from ..frequency.detector import detect_frequency
from ..panel import PanelwiseTransformer, normalize_entity_key
from tsforecast.utils.validation import validate_temporal_data

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
        ...     'column': ['GDP', 'inflation'],
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
            handle_missing_delays: 'ignore', 'ignore', or 'error'
            default_values: Default delay if missing
        """
        # Validation des paramètres
        # Paramètre de stratégie
        if isinstance(strategy, str):
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
        if default_values is not None:
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

        The `fit` method is used to infer transform parameters from data
        and to prepare auxiliary transformers for the application of publication delays.
        It calculates the number of periods to be shifted or masked for each variable according to
        specified delays and prediction date.

        Args:
            X: Time series or panel data. Dates are expected in the index. For panel
                data with a MultiIndex, entities should be on the first n-1 levels
                and dates on the last level.
            y: Ignored.

        Returns:
            self: The fitted transformer instance.
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(self.prediction_date)

        # Inférence des paramètres depuis delays DataFrame si nécessaire
        self.inferred_params_ = self._infer_parameters_from_delays()

        # Construction des dictionnaires de paramètres
        # Fréquence cible (logique spécifique car dépend de la stratégie)
        target_frequency_dict = self._build_target_frequency_dict(X)
        # Unité des délais
        delay_unit_dict = self._build_parameter_dict(
            X=X,
            param_name='delay_unit',
            explicit_value=self.delay_unit,
            inferred_key='delay_unit',
            default_key='delay_unit'
        )
        # Point de référence
        reference_point_dict = self._build_parameter_dict(
            X=X,
            param_name='reference_point',
            explicit_value=self.reference_point,
            inferred_key='reference_point',
            default_key='reference_point'
        )
        
        # Conversion des delays en dictionnaire si DataFrame
        if isinstance(self.delays, pd.DataFrame):
            delays_dict = dict(zip(self.delays['column'], self.delays['applicable_delay']))
        else:
            delays_dict = self.delays

        # Énumération des variables auxquelles appliquer une stratégie de 'shift' et de 'mask'
        if isinstance(self.strategy, str):
            # Distinction suivant la stratégie à appliquer
            if self.strategy == 'shift':
                shift_columns = np.intersect1d(X.columns.tolist(), list(delays_dict.keys())).tolist() if self.default_values is None else X.columns.tolist()
                mask_columns = []
            else:  # équivalent à self.strategy == 'mask'
                mask_columns = np.intersect1d(X.columns.tolist(), list(delays_dict.keys())).tolist() if self.default_values is None else X.columns.tolist()
                shift_columns = []

        else:  # équivalent à isinstance(self.strategy, dict)
            shift_columns = np.intersect1d(X.columns.tolist(), [k for k,v in self.strategy if v == 'shift']).tolist()
            mask_columns = np.intersect1d(X.columns.tolist(), [k for k,v in self.strategy if v == 'mask']).tolist()

        # Détection des fréquences par colonne
        self.detected_frequencies_ = detect_frequency(data=X, time_col=None, panel_cols=None, literal=False, check_consistency=False, strict=False)

        # Calcul du nombre de périodes à shifter pour chaque variable
        # Initialisation du dictionnaire résultat
        self.shift_params = {}
        # Parcours des variables
        for col in shift_columns:
            # Calcul du nombre de périodes à shifter
            n_periods = self._compute_shift_periods(
                col=col,
                delays_dict=delays_dict,
                delay_unit_dict=delay_unit_dict,
                reference_point_dict=reference_point_dict
            )
            # Ajout au dictionnaire résultat
            self.shift_params[col] = {'n_periods': n_periods, 'frequency': self.detected_frequencies_[col]}

        # Calcul du nombre d'observations à masquer pour chaque variable
        # Initialisation du dictionnaire résultat
        self.mask_params = {}
        # Parcours des variables
        for col in mask_columns:
            # Calcul du nombre de périodes à masquer
            result = self._compute_mask_periods(
                col=col,
                X=X,
                delays_dict=delays_dict,
                delay_unit_dict=delay_unit_dict,
                reference_point_dict=reference_point_dict,
                target_frequency_dict=target_frequency_dict
            )
            # Distinction suivant que le masquage est possible ou non
            if result['can_mask']:
                # Ajout au dictionnaire résultat
                self.mask_params[col] = {
                    'n_obs': result['n_periods'],
                    'mask_frequency': result['target_frequency'],
                    'how': 'last'
                }
            else:
                # Warning 
                warnings.warn(f"Could not mask the column '{col}' because it would have created a series of Nan. Moved it to the shifted columns")
                # Ajout au dictionnaire des variables à shift
                self.shift_params[col] = {'n_periods': result['n_periods'], 'frequency': self.detected_frequencies_[col]}

        return self

    # Méthode de transformation des données
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply publication delays to data.

        The `transform` method applies the publication delays to the data using the
        parameters calculated during `fit`. It automatically handles panel data by
        encapsulating auxiliary transformers in a PanelwiseTransformer.

        Args:
            X: Time series or panel data. Dates are expected in the index. For panel
                data with a MultiIndex, entities should be on the first n-1 levels
                and dates on the last level. In the case of panel data, similar
                transformations are applied to all individuals in the panel.

        Returns:
            Transformed data with publication delays applied.
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self)

        # Détection de la structure de panel
        is_panel = isinstance(X.index, pd.MultiIndex)

        # Initialisation du dictionnaire des transformers auxiliaires pour les transformations inverses
        self.auxiliary_transformers_: Dict[str, Dict[tuple, BaseEstimator]] = {'shift': {}, 'mask': {}}

        # Initialisation de la liste des jeux de données résultats
        list_df_transformed = []

        # Traitement des variables à shift
        list_df_transformed.extend(
            self._apply_auxiliary_transformers(
                X=X,
                params_dict=self.shift_params,
                transformer_class=ShiftTransformer,
                transformer_type='shift',
                is_panel=is_panel
            )
        )

        # Traitement des variables à mask
        list_df_transformed.extend(
            self._apply_auxiliary_transformers(
                X=X,
                params_dict=self.mask_params,
                transformer_class=MaskTransformer,
                transformer_type='mask',
                is_panel=is_panel
            )
        )

        # Jointure sur l'index des données transformées
        df_transformed = pd.concat(list_df_transformed, axis=1, join='outer', ignore_index=False)
        # Ajout des colonnes non transformées
        untransformed_columns = set(X.columns) - set(df_transformed.columns)
        if untransformed_columns :
            df_transformed = pd.concat([df_transformed, X[list(untransformed_columns)]], axis=1, join='outer', ignore_index=False)
        # Restauration de l'ordre original des colonnes
        df_transformed = df_transformed[X.columns]

        return df_transformed


    # Méthode de transformation inverse des données
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse publication delay transformation.

        Args:
            X: Transformed data.

        Returns:
            Data with delays reversed.
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self)

        # Initialisation de la liste des jeux de données inversés
        list_df_inversed = []

        # Inversion des shifts
        list_df_inversed.extend(
            self._apply_inverse_transformers(
                X=X,
                params_dict=self.shift_params,
                transformer_type='shift'
            )
        )

        # Inversion des masks
        list_df_inversed.extend(
            self._apply_inverse_transformers(
                X=X,
                params_dict=self.mask_params,
                transformer_type='mask'
            )
        )

        # Jointure sur l'index des données inversées
        df_inversed = pd.concat(list_df_inversed, axis=1, join='outer', ignore_index=False)

        # Ajout des colonnes non transformées
        untransformed_columns = set(X.columns) - set(df_inversed.columns)
        if untransformed_columns:
            df_inversed = pd.concat(
                [df_inversed, X[list(untransformed_columns)]],
                axis=1, join='outer', ignore_index=False
            )

        # Restauration de l'ordre original des colonnes
        df_inversed = df_inversed[X.columns]

        return df_inversed

    # Méthode auxiliaire d'inférence des paramètres d'unité du délai, de point de référence et de fréquence cible
    def _infer_parameters_from_delays(self) -> Dict[str, Any]:
        """Infer delay_unit, reference_point, target_frequency from delays DataFrame.

        Returns:
            Dict of inferred parameters with keys 'delay_unit', 'reference_point',
            and 'target_frequency', each mapping variable names to their values.
        """
        # Initialisation du dictionnaire résultat avec des dictionnaires vides
        inferred = {
            'delay_unit': {},
            'reference_point': {},
            'target_frequency': {}
        }

        # Extraction des éléments du jeu de données
        # Extrait à chaque fois la première valeur en faisant l'hypothèse qu'elle est constante
        if isinstance(self.delays, pd.DataFrame):
            # Inférence de 'delay_unit' à partir de la colonne 'unit'
            if 'unit' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et l'unité
                df_unit = self.delays[['column', 'unit']].drop_duplicates(subset=['column'])
                inferred['delay_unit'] = dict(
                    zip(df_unit['column'], df_unit['unit'])
                )

            # Inférence de 'target_reference_point' à partir de la colonne 'target_reference_point'
            if 'target_reference_point' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et lde point de référence
                df_reference_point = self.delays[['column', 'target_reference_point']].drop_duplicates(subset=['column'])
                inferred['reference_point'] = dict(
                    zip(df_reference_point['column'], df_reference_point['target_reference_point'])
                )

            # Inférence de 'target_frequency' à partir de la colonne 'target_frequency'
            if 'target_frequency' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et la fréquence
                df_frequency = self.delays[['column', 'target_frequency']].drop_duplicates(subset=['column'])
                inferred['target_frequency'] = dict(
                    zip(df_frequency['column'], df_frequency['target_frequency'])
                )

        return inferred

    # Méthode auxiliaire de construction d'un dictionnaire de paramètres
    def _build_parameter_dict(
        self,
        X: pd.DataFrame,
        param_name: str,
        explicit_value: Optional[Union[str, Dict[str, str]]],
        inferred_key: str,
        default_key: str
    ) -> Dict[str, str]:
        """Build a parameter dictionary from inferred, explicit, and default values.

        This method constructs a dictionary mapping column names to parameter values
        by combining (in order of priority): explicit values > inferred values > default values.

        Args:
            X: Input DataFrame to get column names from.
            param_name: Name of the parameter (for warning messages).
            explicit_value: Explicitly provided value (str or dict).
            inferred_key: Key to look up in self.inferred_params_.
            default_key: Key to look up in self.default_values.

        Returns:
            Dictionary mapping column names to parameter values.
        """
        # Initialisation avec les paramètres inférés
        param_dict = self.inferred_params_.get(inferred_key, {}).copy()
        
        # Mise à jour avec les paramètres spécifiés
        if isinstance(explicit_value, dict):
            param_dict.update(explicit_value)
        elif isinstance(explicit_value, str):
            param_dict.update({c: explicit_value for c in X.columns})
        
        # Ajout de la valeur par défaut pour les colonnes restantes
        if self.default_values is not None:
            # Détection des variables qui n'ont pas de valeur
            missing_params = set(X.columns) - set(param_dict.keys())
            
            # Cas où le répertoire des stratégies est un dictionnaire
            if isinstance(self.strategy, dict) and (default_key in self.default_values.keys()):
                missing_params_strategy = missing_params - set(self.strategy.keys())
                if len(missing_params_strategy) > 0:
                    # Ajout de la valeur par défaut
                    for col in missing_params_strategy:
                        param_dict[col] = self.default_values[default_key]
                        warnings.warn(f"Imputed default {param_name} value '{self.default_values[default_key]}' for column '{col}'")
            
            # Cas où la valeur par défaut doit être associée à toutes les colonnes non référencées
            elif (default_key in self.default_values.keys()) and (len(missing_params) > 0):
                for col in missing_params:
                    param_dict[col] = self.default_values[default_key]
                    warnings.warn(f"Imputed default {param_name} value '{self.default_values[default_key]}' for column '{col}'")
            
            # Cas où il y aurait des variables à imputer mais qu'une valeur par défaut n'est pas spécifiée
            elif (default_key not in self.default_values.keys()) and (len(missing_params) > 0):
                warnings.warn(f"Could not impute a default '{param_name}' for columns {missing_params} because it is not specified in the 'default_values' dictionnary")
        else:
            # Détection des variables qui n'ont pas de valeur
            missing_params = set(X.columns) - set(param_dict.keys())
            # Cas où aucune valeur n'est disponible
            warnings.warn(f"Could not impute a default '{param_name}' for columns {missing_params} because it is not specified explicitely, cannot be infered and is not in the 'default_values' dictionnary")
        # Tous les autres cas, absence de valeur par défaut, absence de variable pour laquelle le "param_name" n'est pas spécifiée sont normaux et ne nécessitent ni warning ni imputation
        
        return param_dict

    # Méthode auxiliaire de construction du dictionnaire de fréquence cible
    def _build_target_frequency_dict(self, X: pd.DataFrame) -> Dict[str, str]:
        """Build target frequency dictionary with strategy-aware logic.

        This method constructs a dictionary mapping column names to target frequencies,
        with special handling for the 'mask' strategy which requires target_frequency.

        Args:
            X: Input DataFrame to get column names from.

        Returns:
            Dictionary mapping column names to target frequency values.
        """
        # Initialisation avec les paramètres inférés
        target_frequency_dict = self.inferred_params_.get('target_frequency', {}).copy()
        
        # Mise à jour avec les paramètres spécifiés
        if isinstance(self.target_frequency, dict):
            target_frequency_dict.update(self.target_frequency)
        elif isinstance(self.target_frequency, str):
            target_frequency_dict.update({c: self.target_frequency for c in X.columns})
        
        # Ajout de la valeur par défaut pour les colonnes restantes
        if self.default_values is not None:
            # Détection des variables qui n'ont pas de target frequency
            missing_target_frequency = set(X.columns) - set(target_frequency_dict.keys())
            
            # Si les stratégies sont fournies sous forme de dictionnaire, on vérifie que les variables pour lesquelles la fréquence est manquante sont des 'mask'
            if isinstance(self.strategy, dict) and ('target_frequency' in self.default_values.keys()):
                missing_target_frequency_strategy = missing_target_frequency - set([k for k, v in self.strategy.items() if v == 'shift'])
                if len(missing_target_frequency_strategy) > 0:
                    # Ajout de la valeur par défaut
                    for col in missing_target_frequency_strategy:
                        target_frequency_dict[col] = self.default_values['target_frequency']
                        warnings.warn(f"Imputed default target frequency value '{self.default_values['target_frequency']}' for column '{col}'")
            
            # Cas où toutes les variables doivent être masquées
            elif (self.strategy == "mask") and ('target_frequency' in self.default_values.keys()) and (len(missing_target_frequency) > 0):
                for col in missing_target_frequency:
                    target_frequency_dict[col] = self.default_values['target_frequency']
                    warnings.warn(f"Imputed default target frequency value '{self.default_values['target_frequency']}' for column '{col}'")
            
            # Cas où il y aurait des variables à imputer mais qu'une fréquence par défaut n'est pas spécifiée
            elif ('target_frequency' not in self.default_values.keys()) and (len(missing_target_frequency) > 0):
                # Distinction suivant la stratégie
                if isinstance(self.strategy, dict):
                    missing_target_frequency_strategy = missing_target_frequency - set([k for k, v in self.strategy.items() if v == 'shift'])
                    if len(missing_target_frequency_strategy) > 0:
                        warnings.warn(f"Could not impute a default 'target_frequency' for columns {missing_target_frequency_strategy} because it is not specified in the 'default_values' dictionnary")
                elif self.strategy == "mask":
                    warnings.warn(f"Could not impute a default 'target_frequency' for columns : {missing_target_frequency} because it is not specified in the 'default_values' dictionnary")
        # Tous les autres cas ("shift"), absence de valeur par défaut, absence de variable pour laquelle la "target_frequency" n'est pas spécifiée sont normaux et ne nécessitent ni warning ni imputation
        
        return target_frequency_dict

    # Méthode auxiliaire de calcul du nombre de périodes à shifter
    def _compute_shift_periods(
        self,
        col: str,
        delays_dict: Dict[str, float],
        delay_unit_dict: Dict[str, str],
        reference_point_dict: Dict[str, str]
    ) -> int:
        """Compute the number of periods to shift for a given column.

        This method calculates how many periods of data should be shifted based on
        the publication delay, taking into account the delay unit and reference point.

        Args:
            col: Column name to compute shift periods for.
            delays_dict: Dictionary mapping column names to delay values.
            delay_unit_dict: Dictionary mapping column names to delay units.
            reference_point_dict: Dictionary mapping column names to reference points.

        Returns:
            Number of periods to shift (rounded up to the nearest integer).
        """
        # Normalisation de l'unité des délais
        delay_unit = normalize_duration(delay_unit_dict[col])

        # Calcul des bornes de la période associée à la date de prédiction
        period_start = get_period_start(self.prediction_date_, self.detected_frequencies_[col])
        
        # Calcul du temps écoulé, dans l'unité du délai, entre la date de prédiction et le début de la période
        elapsed_duration = convert_duration(
            value=pd.Timedelta(self.prediction_date_ - period_start).value,
            from_duration='ns',
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

        # Si le point de référence de calcul du délai est la fin, on lui retranche la durée de la période
        if reference_point_dict[col] == 'end':
            elapsed_duration -= period_duration

        # Calcul de l'arrondi à l'unité supérieure de la différence entre le délai et la date de prédiction,
        # divisée par la longueur de la période associée à la fréquence de la série
        n_periods = - math.ceil((delays_dict[col] - elapsed_duration) / period_duration)
        
        return n_periods

    # Méthode auxiliaire de calcul du nombre de périodes à masquer
    def _compute_mask_periods(
        self,
        col: str,
        X: pd.DataFrame,
        delays_dict: Dict[str, float],
        delay_unit_dict: Dict[str, str],
        reference_point_dict: Dict[str, str],
        target_frequency_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """Compute the number of observations to mask for a given column.

        This method calculates how many observations should be masked based on
        the publication delay, and checks if masking is feasible without creating
        an all-NaN series.

        Args:
            col: Column name to compute mask periods for.
            X: Input DataFrame (used to extract index frequency).
            delays_dict: Dictionary mapping column names to delay values.
            delay_unit_dict: Dictionary mapping column names to delay units.
            reference_point_dict: Dictionary mapping column names to reference points.
            target_frequency_dict: Dictionary mapping column names to target frequencies.

        Returns:
            Dictionary with keys:
                - 'n_periods': Number of observations to mask.
                - 'target_frequency': Normalized target frequency.
                - 'can_mask': Boolean indicating if masking is feasible.
        """
        # Normalisation de l'unité des délais
        delay_unit = normalize_duration(delay_unit_dict[col])
        # Normalisation de la fréquence cible
        target_frequency = normalize_frequency(target_frequency_dict[col])

        # Calcul des bornes de la période associée à la date de prédiction
        period_start = get_period_start(self.prediction_date_, self.detected_frequencies_[col])
        
        # Calcul du temps écoulé, dans l'unité du délai, entre la date de prédiction et le début de la période
        elapsed_duration = convert_duration(
            value=pd.Timedelta(self.prediction_date_ - period_start).value,
            from_duration='ns',
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

        # Extraction de la fréquence de l'index
        index_frequency = detect_index_frequency(X.index.get_level_values(-1) if isinstance(X.index, pd.MultiIndex) else X.index)
        
        # Conversion de la durée de la période de l'index dans l'unité du délai
        index_period_duration = convert_duration(
            value=1,
            from_duration=index_frequency,
            to_duration=delay_unit,
            rounding=None
        )

        # Si le point de référence de calcul du délai est la fin, on lui retranche la durée de la période
        if reference_point_dict[col] == 'end':
            elapsed_duration -= period_duration

        # Calcul de l'arrondi à l'unité supérieure de la différence entre le délai et la date de prédiction,
        # divisée par la longueur de la période associée à la fréquence de l'index de la série.
        # Cela donne le nombre de périodes qu'il faut masquer.
        n_periods = math.ceil((delays_dict[col] - elapsed_duration) / index_period_duration)

        # Calcul du nombre d'observations à la fréquence de la série qu'il y a dans la période à la fréquence cible
        target_period_duration = convert_duration(
            value=1,
            from_duration=target_frequency,
            to_duration=self.detected_frequencies_[col],
            rounding=None
        )
        
        # Vérification que le nombre d'observations à masquer est bien strictement inférieur
        # au nombre d'observations dans la période à la 'target_frequency'
        can_mask = math.floor(target_period_duration) > n_periods

        return {
            'n_periods': n_periods,
            'target_frequency': target_frequency,
            'can_mask': can_mask
        }

    # Méthode auxiliaire d'application des transformers auxiliaires
    def _apply_auxiliary_transformers(
        self,
        X: pd.DataFrame,
        params_dict: Dict[str, Dict],
        transformer_class: type,
        transformer_type: str,
        is_panel: bool
    ) -> List[pd.DataFrame]:
        """Apply auxiliary transformers (ShiftTransformer or MaskTransformer) to columns.

        This method groups columns by their transformation parameters and applies
        the appropriate transformer, optionally wrapping in PanelwiseTransformer
        for panel data.

        Args:
            X: Input DataFrame.
            params_dict: Dictionary mapping column names to transformation parameters.
            transformer_class: Class of transformer to use (ShiftTransformer or MaskTransformer).
            transformer_type: Type identifier ('shift' or 'mask') for storage.
            is_panel: Whether the data is panel data (MultiIndex).

        Returns:
            List of transformed DataFrames, one per unique parameter combination.
        """
        # Initialisation de la liste des jeux de données résultats
        list_df_transformed = []
        # Suivi des paramètres déjà traités
        seen_params = set()
        
        # Parcours des colonnes et de leurs paramètres
        for params in params_dict.values():
            # Création d'une clé hashable à partir des paramètres
            params_key = tuple(sorted(params.items()))

            # Vérification si ces paramètres ont déjà été traités
            if params_key not in seen_params:
                # Construction de la liste des colonnes avec ces mêmes paramètres
                columns = [k for k, v in params_dict.items()
                           if tuple(sorted(v.items())) == params_key]

                # Distinction suivant la structure de panel
                if is_panel:
                    # Initialisation d'un PanelwiseTransformer
                    transformer_ = PanelwiseTransformer(
                        transformer=transformer_class(**params),
                        time_col=None,
                        panel_cols=None
                    )
                else:
                    transformer_ = transformer_class(**params)

                # Stockage du transformer
                self.auxiliary_transformers_[transformer_type][params_key] = transformer_

                # Transformation des données
                list_df_transformed.append(transformer_.fit_transform(X[columns]))

                # Marquage des paramètres comme traités
                seen_params.add(params_key)

        return list_df_transformed

    # Méthode auxiliaire d'application des transformations inverses
    def _apply_inverse_transformers(
        self,
        X: pd.DataFrame,
        params_dict: Dict[str, Dict],
        transformer_type: str
    ) -> List[pd.DataFrame]:
        """Apply inverse transformations using stored auxiliary transformers.

        Args:
            X: Transformed DataFrame to inverse.
            params_dict: Dictionary mapping column names to transformation parameters.
            transformer_type: Type identifier ('shift' or 'mask') for retrieval.

        Returns:
            List of inverse-transformed DataFrames, one per unique parameter combination.
        """
        # Initialisation de la liste des jeux de données inversés
        list_df_inversed = []
        # Suivi des paramètres déjà traités
        seen_params = set()
        
        # Parcours des colonnes et de leurs paramètres
        for params in params_dict.values():
            # Création d'une clé hashable à partir des paramètres
            params_key = tuple(sorted(params.items()))

            # Vérification si ces paramètres ont déjà été traités
            if params_key not in seen_params:
                # Récupération du transformer correspondant
                transformer_ = self.auxiliary_transformers_[transformer_type][params_key]

                # Récupération de toutes les colonnes avec ces mêmes paramètres
                columns = [k for k, v in params_dict.items()
                           if tuple(sorted(v.items())) == params_key]

                # Application de la transformation inverse
                list_df_inversed.append(transformer_.inverse_transform(X[columns]))

                # Marquage des paramètres comme traités
                seen_params.add(params_key)

        return list_df_inversed

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
            PublicationDelayTransformer instances.

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


# Fonction auxiliaire de résolution de la stratégie
def _resolve_strategy(
    strategy: Union[str, Dict[Union[tuple, str], str], Callable[[tuple], str]],
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
    # Distinction suivant le type de la stratégie
    if isinstance(strategy, str):
        # Stratégie globale
        if strategy not in ('shift', 'mask'):
            raise ValueError(f"Invalid strategy: '{strategy}'. Must be 'shift' or 'mask'.")
        return strategy

    elif isinstance(strategy, dict):
        # Dictionnaire de stratégies par entité
        if (entity_key not in strategy):
            # Tentative avec clé non-tuple si entité simple
            if len(entity_key) == 1 and (entity_key[0] in strategy):
                return strategy[entity_key[0]]
            # Si toutes les clés sont des strings et non des tuples, alors il s'agit d'un dictionnaire qui associe à chaque variable une stratégie, quelle que soit l'entité
            elif all([isinstance(k, str) for k in strategy.keys()]) :
                return strategy
            else :
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

# Fonction de construction des kwargs pour chaque transformer spécifique à une entité
def prepare_entity_kwargs_from_delays(
    df_delays: pd.DataFrame,
    strategy: Union[
        Literal['shift', 'mask'],
        Dict[Union[tuple, str], Literal['shift', 'mask']]
    ] = 'shift',
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

    # Construction des configs
    entity_params = _build_entity_params(
        df_delays=df_delays,
        delay_col=delay_col,
        unit_col=unit_col,
        reference_point_col=reference_point_col,
        target_frequency_col=target_frequency_col
    )

    # Conversion au format des entity_kwargs 
    # Initialisation du dictionnaire résultat
    entity_kwargs = {}
    # Parcours des paramètres
    for entity_key, params in entity_params.items():
        # Résolution de la stratégie
        entity_strategy = _resolve_strategy(strategy=strategy, entity_key=entity_key)

        # Construction des kwargs
        entity_kwargs[entity_key] = {
            'delays': params['delays'],
            'delay_unit': params['delay_unit'],
            'reference_point': params['reference_point'],
            'target_frequency': params['target_frequency'],
            'strategy': entity_strategy
        }

    return entity_kwargs


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
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_index_frequency(X.index)

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
        self.index_frequency_, self.index_position_, self.index_suffix_ = detect_and_parse_index_frequency(X.index)
    
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
        mask_frequency, setting masked values to NaN. For incomplete periods
        at boundaries, artificially extends the period before masking, then
        returns to original index. Stores masked rows in self.masked_data_ 
        for later restoration via inverse_transform.

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
        
        # Stockage de l'index original pour filtrer les résultats à la fin
        original_index = data.index

        # Génération des périodes
        periods = self._generate_periods(
            start_date=data.index.min(),
            end_date=data.index.max(),
            frequency=self.mask_frequency
        )

        # Liste pour collecter les lignes masquées
        masked_rows_list = []

        # Masque dans chaque période
        for i, (period_start, period_end) in enumerate(periods):
            # Détermination si c'est la première ou dernière période
            is_first_period = (i == 0)
            is_last_period = (i == len(periods) - 1)
            
            # Filtre des observations dans cette période
            period_mask = (data.index >= period_start) & (data.index < period_end)
            period_obs = data[period_mask]
            
            # Extension artificielle pour les périodes incomplètes aux extrémités
            if (is_first_period or is_last_period) and len(period_obs) > 0:
                period_obs = self._extend_period_if_incomplete(
                    period_obs, 
                    period_start, 
                    period_end,
                    is_first_period,
                    is_last_period
                )

            # Calcul du nombre de périodes à masquer
            n_to_mask = min(self.n_obs, len(period_obs))

            if (len(period_obs) > 0) & (self.how == 'last'):
                # Masque des n_obs plus récentes
                most_recent_indices = period_obs.index[-n_to_mask:]
                # Stockage uniquement des lignes qui existaient dans les données originales
                original_most_recent = [idx for idx in most_recent_indices if idx in original_index]
                if original_most_recent:
                    masked_rows_list.append(data.loc[original_most_recent])
                # Masquage dans masked_data (seulement les indices originaux)
                for idx in most_recent_indices:
                    if idx in masked_data.index:
                        masked_data.loc[idx] = np.nan
                        
            elif (len(period_obs) > 0) & (self.how =='first'):
                # Masque des n_obs les plus anciennes
                oldest_indices = period_obs.index[:n_to_mask]
                # Stockage uniquement des lignes qui existaient dans les données originales
                original_oldest = [idx for idx in oldest_indices if idx in original_index]
                if original_oldest:
                    masked_rows_list.append(data.loc[original_oldest])
                # Masquage dans masked_data (seulement les indices originaux)
                for idx in oldest_indices:
                    if idx in masked_data.index:
                        masked_data.loc[idx] = np.nan

        # Concaténation de toutes les lignes masquées dans self.masked_data_
        if masked_rows_list:
            self.masked_data_ = pd.concat(masked_rows_list)
        else:
            # Aucune ligne masquée
            if isinstance(data, pd.Series):
                self.masked_data_ = pd.Series(dtype=data.dtype, name=data.name)
            else:
                self.masked_data_ = pd.DataFrame(columns=data.columns)

        return masked_data

    # Méthode auxiliaire d'extension des périodes incomplètes
    def _extend_period_if_incomplete(
        self,
        period_obs: Union[pd.Series, pd.DataFrame],
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        is_first_period: bool,
        is_last_period: bool
    ) -> Union[pd.Series, pd.DataFrame]:
        """Extend incomplete periods at boundaries with artificial dates.
        
        For incomplete periods at the start or end of the data, this method
        adds missing sub-periods (filled with NaN) to ensure proper masking
        behavior. After masking, only the original indices are kept.
        
        Args:
            period_obs: Observations in the current period
            period_start: Start of the period
            period_end: End of the period
            is_first_period: Whether this is the first period
            is_last_period: Whether this is the last period
            
        Returns:
            Extended period observations (or original if already complete)
        """
        # Reconstitution de la fréquence avec position et suffixe
        pandas_freq = build_frequency_string(
            self.index_frequency_,
            self.index_position_,
            self.index_suffix_
        )
        
        # Génération de l'index complet pour cette période
        full_period_index = pd.date_range(
            start=period_start,
            end=period_end,
            freq=pandas_freq,
            inclusive='left'  # Exclusion de period_end
        )
        
        # Vérification si la période est déjà complète
        if len(period_obs) == len(full_period_index):
            return period_obs
        
        # Extension seulement aux extrémités
        first_obs_date = period_obs.index.min()
        last_obs_date = period_obs.index.max()
        
        # Vérification que c'est bien une période incomplète aux extrémités
        if is_first_period:
            # Période incomplète au début : vérification si les premières dates manquent
            if first_obs_date > period_start:
                # Création des dates manquantes au début
                missing_start_dates = full_period_index[full_period_index < first_obs_date]
                if len(missing_start_dates) > 0:
                    # Création d'une série/dataframe avec NaN pour les dates manquantes
                    if isinstance(period_obs, pd.Series):
                        missing_data = pd.Series(
                            np.nan, 
                            index=missing_start_dates,
                            name=period_obs.name,
                            dtype=period_obs.dtype
                        )
                    else:
                        missing_data = pd.DataFrame(
                            index=missing_start_dates,
                            columns=period_obs.columns
                        )
                    # Concaténation avec les données existantes et tri
                    period_obs = pd.concat([missing_data, period_obs]).sort_index()
        
        if is_last_period:
            # Période incomplète à la fin : vérification si les dernières dates manquent
            if last_obs_date < full_period_index[-1]:
                # Création des dates manquantes à la fin
                missing_end_dates = full_period_index[full_period_index > last_obs_date]
                if len(missing_end_dates) > 0:
                    # Création d'une série/dataframe avec NaN pour les dates manquantes
                    if isinstance(period_obs, pd.Series):
                        missing_data = pd.Series(
                            np.nan,
                            index=missing_end_dates,
                            name=period_obs.name,
                            dtype=period_obs.dtype
                        )
                    else:
                        missing_data = pd.DataFrame(
                            index=missing_end_dates,
                            columns=period_obs.columns
                        )
                    # Concaténation avec les données existantes et tri
                    period_obs = pd.concat([period_obs, missing_data]).sort_index()
        
        return period_obs

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
            # Ajout du tuple à la liste
            periods.append((period_start, period_end))

        return periods