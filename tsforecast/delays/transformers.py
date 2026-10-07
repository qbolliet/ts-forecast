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
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick
import math
from typing import Any, Callable, Dict, Optional, Union, List, Literal, Tuple
from datetime import datetime
import logging
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Importation des modules du package
from tsforecast.utils.frequency import (
    normalize_frequency,
    is_higher_frequency,
    detect_index_frequency,
    detect_frequency,
)
from tsforecast.utils.time import resolve_date, get_period_start, get_period_boundaries
from tsforecast.utils.duration import convert_duration, normalize_duration, DurationConverter
from ..panel import PanelwiseTransformer, normalize_entity_key, is_panel_data, get_entity_levels
from tsforecast.utils.validation import validate_temporal_data
from tsforecast.utils.parse import build_frequency_string
from tsforecast.utils._constants import BUSINESS_DAYS_PER_WEEK, DAYS_PER_WEEK
from .report import DelayFitReport, ColumnDelayRecord

# Journalisation : aucun handler n'est configuré ici, c'est à l'application d'en fournir un
logger = logging.getLogger(__name__)

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
        fit_report_: :class:`~tsforecast.delays.DelayFitReport` of the last ``fit``: resolved
            setting of each column and its origin, columns ignored or unaffected, defaults
            imputed, mask-to-shift fallbacks

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
        ...     'frequency': ['M', 'M']
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
            delays_dict = dict(zip(self.delays['column'], self.delays['delay']))
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

        # Détection des fréquences par colonne (return_format='base' par défaut)
        self.detected_frequencies_ = detect_frequency(data=X, time_col=None, panel_cols=None, check_consistency=False, strict=False)

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
        # Initialisation du dictionnaire résultat et de la liste des variables masquables seulement par décalage
        self.mask_params = {}
        mask_fallbacks: List[str] = []
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
                # Un nombre négatif signifie une donnée déjà publiée : rien à masquer
                self.mask_params[col] = {
                    'n_obs': max(0, result['n_periods']),
                    'mask_frequency': result['target_frequency'],
                    'how': 'last'
                }
            else:
                # Warning 
                warnings.warn(f"Could not mask the column '{col}' because it would have created a series of Nan. Moved it to the shifted columns")
                # Ajout au dictionnaire des variables à shift
                self.shift_params[col] = {'n_periods': result['n_periods'], 'frequency': self.detected_frequencies_[col]}
                mask_fallbacks.append(col)

        # Rapport d'ajustement : tout ce que le fit a résolu, sans avertissement à relire
        self.fit_report_ = self._build_fit_report(
            X=X,
            delays_dict=delays_dict,
            delay_unit_dict=delay_unit_dict,
            reference_point_dict=reference_point_dict,
            target_frequency_dict=target_frequency_dict,
            mask_fallbacks=mask_fallbacks
        )
        # Logging
        logger.info(self.fit_report_.summary())

        return self

    # Méthode auxiliaire de détermination de l'origine d'un paramètre
    def _parameter_source(
        self,
        col: str,
        explicit_value: Optional[Union[str, Dict[str, str]]],
        inferred_key: str,
        default_key: str,
        resolved: Dict[str, str]
    ) -> Optional[str]:
        """Tell where the resolved value of a parameter comes from for a column.

        Follows the priority of ``_build_parameter_dict``: explicit > inferred > default.

        Args:
            col: Column name.
            explicit_value: Value given to the constructor (str, dict or None).
            inferred_key: Key of the parameter in ``inferred_params_``.
            default_key: Key of the parameter in ``default_values``.
            resolved: Resolved parameter dictionary (column -> value).

        Returns:
            'explicit', 'inferred', 'default', or None if the column has no resolved value.
        """
        # Colonne sans valeur résolue
        if col not in resolved:
            return None
        # Valeur passée au constructeur, par colonne ou pour toutes les colonnes
        if isinstance(explicit_value, str) or (isinstance(explicit_value, dict) and col in explicit_value):
            return 'explicit'
        # Valeur déduite du DataFrame des délais
        if col in self.inferred_params_.get(inferred_key, {}):
            return 'inferred'
        # Valeur par défaut, seule origine restante
        if self.default_values is not None and default_key in self.default_values:
            return 'default'
        return None

    # Méthode auxiliaire de construction du rapport d'ajustement
    def _build_fit_report(
        self,
        X: pd.DataFrame,
        delays_dict: Dict[str, float],
        delay_unit_dict: Dict[str, str],
        reference_point_dict: Dict[str, str],
        target_frequency_dict: Dict[str, str],
        mask_fallbacks: List[str]
    ) -> DelayFitReport:
        """Build the :class:`DelayFitReport` of the fit from the resolved parameters.

        Args:
            X: Data the transformer was fitted on.
            delays_dict: Delay of each variable.
            delay_unit_dict: Resolved delay unit of each variable.
            reference_point_dict: Resolved reference point of each variable.
            target_frequency_dict: Resolved target frequency of each variable.
            mask_fallbacks: Variables moved from mask to shift.

        Returns:
            The immutable fit report.
        """
        # Une ligne par variable retardée, 'shift' d'abord puis 'mask' (ordre de l'application)
        records = []
        for strategy, params_dict in (('shift', self.shift_params), ('mask', self.mask_params)):
            for col, params in params_dict.items():
                is_mask = strategy == 'mask'
                records.append(ColumnDelayRecord(
                    column=col,
                    strategy=strategy,
                    delay=delays_dict.get(col),
                    delay_unit=delay_unit_dict.get(col),
                    reference_point=reference_point_dict.get(col),
                    frequency=params.get('frequency') if not is_mask else self.detected_frequencies_.get(col),
                    n_periods=None if is_mask else params['n_periods'],
                    n_obs=params['n_obs'] if is_mask else None,
                    target_frequency=params['mask_frequency'] if is_mask else None,
                    delay_unit_source=self._parameter_source(
                        col, self.delay_unit, 'delay_unit', 'delay_unit', delay_unit_dict),
                    reference_point_source=self._parameter_source(
                        col, self.reference_point, 'reference_point', 'reference_point', reference_point_dict),
                    target_frequency_source=(
                        self._parameter_source(
                            col, self.target_frequency, 'target_frequency', 'target_frequency', target_frequency_dict)
                        if is_mask else None
                    ),
                    moved_from_mask=(not is_mask) and (col in mask_fallbacks)
                ))

        # Couples (variable, paramètre) complétés par les valeurs par défaut
        defaults_imputed = tuple(
            (record.column, name)
            for record in records
            for name in ('delay_unit', 'reference_point', 'target_frequency')
            if getattr(record, f'{name}_source') == 'default'
        )

        # Variables de X sans délai, et variables de la spécification des délais absentes de X
        delayed = {record.column for record in records}
        return DelayFitReport(
            prediction_date=self.prediction_date_,
            columns=tuple(records),
            columns_unaffected=tuple(col for col in X.columns if col not in delayed),
            columns_ignored=tuple(col for col in delays_dict if col not in X.columns),
            defaults_imputed=defaults_imputed,
            mask_fallbacks=tuple(mask_fallbacks)
        )

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
        is_panel = is_panel_data(X)

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

        # Jointure sur l'index des données transformées (aucune colonne transformée : index seul)
        df_transformed = (pd.concat(list_df_transformed, axis=1, join='outer', ignore_index=False)
                          if list_df_transformed else pd.DataFrame(index=X.index))
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

        # Jointure sur l'index des données inversées (aucune colonne inversée : index seul)
        df_inversed = (pd.concat(list_df_inversed, axis=1, join='outer', ignore_index=False)
                       if list_df_inversed else pd.DataFrame(index=X.index))

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

            # Inférence de 'reference_point' à partir de la colonne 'reference_point'
            if 'reference_point' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et le point de référence
                df_reference_point = self.delays[['column', 'reference_point']].drop_duplicates(subset=['column'])
                inferred['reference_point'] = dict(
                    zip(df_reference_point['column'], df_reference_point['reference_point'])
                )

            # Inférence de 'target_frequency' à partir de la colonne 'frequency'
            if 'frequency' in self.delays.columns:
                # Stockage sous la forme d'un dictionnaire de l'association entre les variables et la fréquence
                df_frequency = self.delays[['column', 'frequency']].drop_duplicates(subset=['column'])
                inferred['target_frequency'] = dict(
                    zip(df_frequency['column'], df_frequency['frequency'])
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
        # Les masques nuls ne transforment rien : colonnes laissées telles quelles
        params_dict = _active_params(params_dict, transformer_type)

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
        # Les masques nuls n'ont pas été appliqués
        params_dict = _active_params(params_dict, transformer_type)

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

# Fonction de filtrage des paramètres sans effet
def _active_params(params_dict: Dict[str, Dict], transformer_type: str) -> Dict[str, Dict]:
    """Drop the mask parameters that mask nothing (``n_obs == 0``).

    A zero mask is a no-op: no ``MaskTransformer`` is built for it, which also
    spares the check of its mask frequency (equal to the index frequency when
    the delay is shorter than the elapsed part of the period).

    Args:
        params_dict: Dictionary mapping column names to transformation parameters.
        transformer_type: ``'shift'`` or ``'mask'``.

    Returns:
        The parameters to apply.

    Examples:
        >>> _active_params({'a': {'n_obs': 0}, 'b': {'n_obs': 2}}, 'mask')
        {'b': {'n_obs': 2}}
    """
    # Retourne les paramètres intacts pour le mode 'shift'
    if transformer_type != 'mask':
        return params_dict
    # Filtre des colonnes pour lesquelles le nombre d'observations masquées est non nul
    return {col: params for col, params in params_dict.items() if params['n_obs'] != 0}


# Fonction de détection des composantes de la fréquence d'un index
def _detect_index_components(index: pd.Index) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect the base frequency, position and anchor of an index.

    The delay transformers shift and mask by whole index periods, which
    assumes one period per index step: a multiplied frequency ('2MS')
    is rejected rather than treated as its base.

    Args:
        index: Time index of the data.

    Returns:
        Tuple ``(base, position, suffix)`` of the index frequency.

    Raises:
        ValueError: If no frequency can be detected or if it carries a
            multiplier.

    Examples:
        >>> _detect_index_components(pd.date_range('2024-01-01', periods=6, freq='QS'))
        ('Q', 'S', 'JAN')
    """
    # Détection de la fréquence de l'index
    parsed = detect_index_frequency(index, return_format='components')
    # Cas où la fréquence n'a pas pu être détectée
    if parsed is None:
        raise ValueError("Could not detect index frequency. Index may be irregular or have insufficient observations.")
    # Les index multipliés ne sont pas supportés
    if parsed.multiplier != 1:
        raise ValueError(
            f"Multiplied index frequency is not supported by the delay transformers "
            f"(detected multiplier {parsed.multiplier}). Resample the data to a "
            f"regular one-period frequency first."
        )
    return parsed.freq, parsed.position, parsed.suffix


def create_delay_transformer_factory(
    df_delays: pd.DataFrame,
    strategy: Union[
        Literal['shift', 'mask'],
        Dict[Union[str, tuple], Literal['shift', 'mask']],
        Callable[[tuple], Literal['shift', 'mask']]
    ] = 'shift',
    prediction_date: Union[str, datetime] = 'today',
    delay_col: str = 'delay',
    unit_col: str = 'unit',
    reference_point_col: str = 'reference_point',
    target_frequency_col: str = 'frequency',
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
        delay_col: Column name for delay values. Defaults to 'delay'.
        unit_col: Column name for delay units. Defaults to 'unit'.
        reference_point_col: Column name for reference point.
            Defaults to 'reference_point'.
        target_frequency_col: Column name for target frequency.
            Defaults to 'frequency'.
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
    for entity_key, group in df_delays.groupby(level=get_entity_levels(df_delays)):
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
    delay_col: str = 'delay',
    unit_col: str = 'unit',
    reference_point_col: str = 'reference_point',
    target_frequency_col: str = 'frequency'
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


# Nom de colonne interne d'une Series convertie en DataFrame
_SERIES_COLUMN = '__series__'


# Fonction de validation d'un paramètre entier
def _validate_integer(value: Any, name: str, allow_negative: bool) -> None:
    """Check that a parameter is an integer, optionally non-negative.

    Args:
        value: Value of the parameter.
        name: Name of the parameter, used in the error messages.
        allow_negative: Whether negative values are accepted.

    Raises:
        TypeError: If ``value`` is not an integer (booleans are rejected).
        ValueError: If ``value`` is negative while ``allow_negative`` is False.

    Examples:
        >>> _validate_integer(3, 'n_obs', allow_negative=False)
        >>> _validate_integer(-1, 'n_obs', allow_negative=False)
        Traceback (most recent call last):
        ...
        ValueError: 'n_obs' must be a non-negative integer, got -1
    """
    # Les booléens sont des entiers pour Python, mais pas un nombre de périodes valide
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"'{name}' must be an integer, got {type(value).__name__}")
    if not allow_negative and value < 0:
        raise ValueError(f"'{name}' must be a non-negative integer, got {value}")


# Fonction de validation des données d'entrée des transformateurs auxiliaires
def _validate_time_series_input(
    X: Union[pd.Series, pd.DataFrame],
    owner: str
) -> Tuple[Union[pd.Series, pd.DataFrame], Optional[str]]:
    """Validate a time series and return it sorted, on a ``DatetimeIndex``.

    Args:
        X: Series or DataFrame indexed by dates (``DatetimeIndex``, ``PeriodIndex``
            or date strings).
        owner: Name of the calling class, used in the error messages.

    Returns:
        Tuple ``(data, period_freq)``: the validated data, sorted by date, and the
        frequency of the input ``PeriodIndex`` (None for any other index), used to
        give the output the type of the input index.

    Raises:
        ValueError: If ``X`` is not a pandas object, has a ``MultiIndex`` (panel
            data), or has an index that cannot be read as dates or with duplicates.

    Examples:
        >>> s = pd.Series([1.0, 2.0], index=pd.period_range('2024-01', periods=2, freq='M'))
        >>> data, period_freq = _validate_time_series_input(s, 'ShiftTransformer')
        >>> type(data.index).__name__, period_freq
        ('DatetimeIndex', 'M')
    """
    # Validation du type de données
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise ValueError("X must be a pandas Series or DataFrame")
    # Les panels passent par PanelwiseTransformer, entité par entité
    if isinstance(X.index, pd.MultiIndex):
        raise ValueError(
            f"{owner} expects a single time index, got a MultiIndex (panel data): "
            f"wrap it in a PanelwiseTransformer to apply it entity by entity."
        )
    # Mémorisation du type d'index périodique pour la restitution
    period_freq = X.index.freqstr if isinstance(X.index, pd.PeriodIndex) else None
    # Validation de la structure temporelle et tri
    data = validate_temporal_data(data=X, time_col=None, panel_cols=None, strict=True, sort_data=True, return_metadata=False)
    return data, period_freq


# Fonction de restitution du type d'index d'entrée
def _restore_index_type(
    data: Union[pd.Series, pd.DataFrame],
    period_freq: Optional[str]
) -> Union[pd.Series, pd.DataFrame]:
    """Give back a ``PeriodIndex`` to data whose input index was periodic.

    Args:
        data: Data on a ``DatetimeIndex`` (period starts).
        period_freq: Frequency of the input ``PeriodIndex``, None if the input
            index was not periodic.

    Returns:
        ``data`` itself, its index converted to periods when ``period_freq`` is set.
    """
    # Cas où l'index est un period index
    if period_freq is not None:
        data.index = data.index.to_period(period_freq)
    # Cas où l'index est un datetime index
    return data


# Fonction de détection de l'offset de la grille de l'index
def _index_offset(base: str, position: Optional[str], suffix: Optional[str]) -> pd.DateOffset:
    """Build the pandas offset of the grid of an index from its components.

    Args:
        base: Base frequency code ('M', 'Q', 'D', 'W', 'h', ...).
        position: Position ('S', 'E' or None).
        suffix: Anchor ('DEC', 'WED', ... or None).

    Returns:
        The offset of one index period (period end when no position is known).

    Examples:
        >>> _index_offset('Q', 'E', 'DEC').freqstr
        'QE-DEC'
    """
    return to_offset(build_frequency_string(base, position, suffix, default_position='E'))


# Transformer décalant les séries d'un nombre donné de périodes
class ShiftTransformer(BaseEstimator, TransformerMixin):
    """Shift time series data by a number of calendar periods, without data loss.

    Every date moves by exactly ``n_periods`` periods of ``frequency``, converted
    into periods of the index frequency detected at ``fit``: the values are kept,
    only their dates change. A **positive** ``n_periods`` moves the values to
    **earlier** dates, a negative one to later dates. ``PublicationDelayTransformer``
    passes a negative ``n_periods``, so that the value observed at ``t`` appears at
    ``t + delay``.

    The shift is calendar arithmetic on each date (``date - n * offset``): it does
    not depend on the neighbouring dates, so an index with gaps (or an irregular
    index whose dates lie on the grid of the detected frequency) keeps its gaps,
    and ``inverse_transform`` restores the input exactly.

    When ``frequency`` is coarser than the index, the number of index periods is
    ``round(n_periods * factor)``, with ``factor`` the nominal ratio of the two
    durations (1 month = 30 days, 1 week = 7 days, 1 quarter = 3 months; on a
    business-day index, 1 week = 5 business days). The conversion is exact for
    nested calendar units (quarters on a monthly index, years on a quarterly one,
    weeks on a daily one) and approximate otherwise (months on a daily index).
    A multiplier of ``frequency`` is honoured (``'2M'`` = two months per period).

    Args:
        n_periods: Number of periods of ``frequency`` to shift; positive values
            move the data to earlier dates, negative values to later dates.
        frequency: Frequency of the shift arithmetic ('D', 'W', 'M', 'Q', 'Y',
            'h', ...), possibly multiplied; it cannot be finer than the index.

    Attributes:
        index_frequency_: Base code of the index frequency detected at ``fit``.
        index_position_: Position ('S', 'E' or None) of the index frequency.
        index_suffix_: Anchor of the index frequency (e.g. 'DEC', 'WED') or None.
        index_offset_: Pandas offset of one index period.
        index_periods_: Number of index periods the dates move by in ``transform``.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        >>> series = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates, name='GDP')
        >>> shifter = ShiftTransformer(n_periods=-2, frequency='M')
        >>> shifted = shifter.fit_transform(series)
        >>> shifted.index[0].strftime('%Y-%m-%d')  # value of January published in March
        '2024-03-01'
        >>> shifter.inverse_transform(shifted).equals(series)
        True
    """

    # Initialisation
    def __init__(
        self,
        n_periods: int,
        frequency: str,
    ):
        """Initialize ShiftTransformer.

        Args:
            n_periods: Number of periods to shift (positive = earlier dates).
            frequency: Frequency of the shift arithmetic ('D', 'M', 'Q', ...).
        """
        # Initialisation des attributs
        self.n_periods = n_periods
        self.frequency = frequency

    # Méthode d'entraînement
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Validate the parameters and detect the index frequency.

        Args:
            X: Time series or DataFrame indexed by dates.
            y: Ignored.

        Returns:
            self

        Raises:
            TypeError: If ``n_periods`` is not an integer.
            ValueError: If ``X`` is not a pandas Series / DataFrame, has a
                ``MultiIndex``, an index that cannot be read as dates, duplicated
                dates, fewer than two observations, an undetectable or multiplied
                frequency, or if ``frequency`` is unknown or finer than the index.
        """
        # Validation des paramètres
        _validate_integer(self.n_periods, 'n_periods', allow_negative=True)

        # Validation de la structure temporelle des données
        data, _ = _validate_time_series_input(X, 'ShiftTransformer')

        # Détection de la fréquence de l'index
        base, position, suffix = _detect_index_components(data.index)

        # Validation : la fréquence du décalage ne doit pas être plus fine que l'index
        if is_higher_frequency(self.frequency, base):
            raise ValueError(
                f"Shift frequency '{self.frequency}' cannot be more granular "
                f"than index frequency '{base}'. "
                f"Example: you can shift by months ('M') on a daily ('D') index, "
                f"but not by days ('D') on a monthly ('M') index."
            )

        # Stockage des composantes de l'index et du décalage en périodes d'index
        self.index_frequency_, self.index_position_, self.index_suffix_ = base, position, suffix
        self.index_offset_ = _index_offset(base, position, suffix)
        self.index_periods_ = self._to_index_periods(base)

        return self

    # Méthode de transformation
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Move every date by ``n_periods`` periods (earlier for a positive shift).

        Args:
            X: Time series or DataFrame indexed by dates on the grid of the index
                frequency detected at ``fit`` (a single observation is enough).

        Returns:
            Shifted data of the same type, sorted by date, with the same values,
            columns, dtypes, names and index type.

        Raises:
            NotFittedError: If the transformer is not fitted.
            ValueError: If ``X`` is invalid (see ``fit``) or has dates off the grid
                of the index frequency.
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self, 'index_offset_')

        # Validation de la structure temporelle des données
        data, period_freq = _validate_time_series_input(X, 'ShiftTransformer')

        # Décalage vers le passé pour un nombre de périodes positif
        return _restore_index_type(self._shift(data, self.index_periods_), period_freq)

    # Méthode d'inversion de la transformation
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Move every date back by ``n_periods`` periods.

        ``inverse_transform(transform(X))`` restores ``X`` exactly (sorted by date).

        Args:
            X: Shifted time series or DataFrame.

        Returns:
            Data with its original dates.

        Raises:
            NotFittedError: If the transformer is not fitted.
            ValueError: If ``X`` is invalid (see ``fit``) or has dates off the grid
                of the index frequency.
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self, 'index_offset_')

        # Validation de la structure temporelle des données
        data, period_freq = _validate_time_series_input(X, 'ShiftTransformer')

        # Décalage opposé
        return _restore_index_type(self._shift(data, -self.index_periods_), period_freq)

    # Méthode auxiliaire de conversion du décalage en périodes d'index
    def _to_index_periods(self, index_base: str) -> int:
        """Convert ``n_periods`` periods of ``frequency`` into index periods.

        Args:
            index_base: Base code of the index frequency.

        Returns:
            Number of index periods, rounded to the nearest integer.
        """
        # Conversion via le facteur nominal entre durées (multiplicateur inclus)
        converter = DurationConverter()
        if index_base == 'B' and normalize_frequency(self.frequency) != 'B':
            # Index en jours ouvrés : durée calendaire ramenée à 5 jours ouvrés par semaine
            factor = converter.get_conversion_factor(self.frequency, 'D') * BUSINESS_DAYS_PER_WEEK / DAYS_PER_WEEK
        else:
            factor = converter.get_conversion_factor(self.frequency, index_base)

        return int(round(self.n_periods * factor))

    # Méthode auxiliaire de décalage calendaire
    def _shift(self, data: Union[pd.Series, pd.DataFrame], index_periods: int) -> Union[pd.Series, pd.DataFrame]:
        """Move every date of ``data`` back by ``index_periods`` index periods.

        Args:
            data: Validated data (sorted, ``DatetimeIndex``).
            index_periods: Number of index periods (positive = earlier dates).

        Returns:
            Copy of ``data`` on the shifted index.

        Raises:
            ValueError: If a date is off the grid of the index frequency.
        """
        # Copie indépendante des données
        result = data.copy()
        # Cas où aucun décalage n'est nécessaire
        if index_periods == 0:
            return result

        # Vérification que les dates sont sur la grille (arithmétique d'offset exacte et réversible)
        self._check_on_grid(data.index)

        # Décalage calendaire de chaque date, indépendamment de ses voisines
        result.index = data.index - index_periods * self.index_offset_
        return result

    # Méthode auxiliaire de vérification de l'alignement des dates sur la grille de l'index
    def _check_on_grid(self, index: pd.DatetimeIndex) -> None:
        """Check that every date lies on the grid of the index frequency.

        Fixed-length offsets (days, hours, ...) are exact on any date; calendar
        offsets (month starts, quarter ends, Wednesdays, business days...) roll a
        date that is not on their grid, which would break the inversion.

        Args:
            index: Dates to check.

        Raises:
            ValueError: If a date is off the grid.
        """
        # Les offsets de durée fixe sont exacts sur toute date
        if isinstance(self.index_offset_, Tick):
            return
        off_grid = [date for date in index if not self.index_offset_.is_on_offset(date)]
        if off_grid:
            raise ValueError(
                f"ShiftTransformer shifts by whole periods of the index frequency "
                f"'{self.index_offset_.freqstr}', but {len(off_grid)} date(s) are not on its grid "
                f"(first: {off_grid[0]}). Align the index on the frequency first."
            )


# Transformer masquant un nombre donné d'observations par période
class MaskTransformer(BaseEstimator, TransformerMixin):
    """Mask the first or last ``n_obs`` positions of each period.

    The index is cut into the calendar periods of ``mask_frequency`` (anchors and
    multipliers honoured: ``'QE-NOV'``, ``'2Q'``...). In each period, the positions
    are those of the **regular grid** of the index frequency detected at ``fit``
    (e.g. the 31 days of January for a daily index, the 3 months of a quarter for
    a monthly one), whatever the dates present: ``how='last'`` masks the
    observations on the last ``n_obs`` positions, ``how='first'`` those on the
    first ones. A position absent from the data (start or end of the series, gap,
    irregular index) masks nothing, and an observation is never masked because
    a neighbouring one is missing. This reproduces, in every period of the
    history, the information missing at the same position of the period as the
    prediction date (``PublicationDelayTransformer``, strategy ``'mask'``).

    Masked cells are set to NaN and stored; ``inverse_transform`` puts their
    original values back. The store accumulates the cells masked by every
    ``transform`` call since ``fit`` (the latest value wins for a date masked
    twice), so that the train and test sets transformed by the same fitted
    instance can both be inverted.

    Args:
        n_obs: Number of positions to mask per period (non-negative integer).
        mask_frequency: Frequency of the periods ('W', 'M', 'Q', 'Y', '2Q', ...),
            strictly coarser than the index frequency.
        how: ``'last'`` (default) masks the last positions of each period,
            ``'first'`` the first ones.

    Attributes:
        index_frequency_: Base code of the index frequency detected at ``fit``.
        index_position_: Position ('S', 'E' or None) of the index frequency.
        index_suffix_: Anchor of the index frequency or None.
        index_offset_: Pandas offset of one index period (grid of the positions).
        masked_values_: Dictionary ``{column: Series}`` of the original values of
            the masked cells (a Series is stored under an internal column name).
        original_dtypes_: Dictionary ``{column: dtype}`` of the transformed data,
            used to give integer columns back their dtype at the inversion.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS')
        >>> series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates, name='GDP')
        >>> masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        >>> masked = masker.fit_transform(series)
        >>> masked[masked.isna()].index.strftime('%Y-%m').tolist()  # last month of each quarter
        ['2024-03', '2024-06']
        >>> masker.inverse_transform(masked).equals(series)
        True
    """

    # Initialisation
    def __init__(self, n_obs: int, mask_frequency: str, how: Literal['first', 'last'] = 'last'):
        """Initialize MaskTransformer.

        Args:
            n_obs: Number of positions to mask per period.
            mask_frequency: Frequency of the periods ('W', 'M', 'Q', ...).
            how: ``'first'`` or ``'last'`` positions of each period.
        """
        # Initialisation des attributs (aucun attribut ajusté avant fit, convention sklearn)
        self.n_obs = n_obs
        self.mask_frequency = mask_frequency
        self.how = how

    # Méthode d'entraînement
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Validate the parameters, detect the index frequency and empty the store.

        Args:
            X: Time series or DataFrame indexed by dates.
            y: Ignored.

        Returns:
            self

        Raises:
            TypeError: If ``n_obs`` is not an integer.
            ValueError: If ``n_obs`` is negative, ``how`` is not ``'first'`` or
                ``'last'``, ``mask_frequency`` is unknown or not strictly coarser
                than the index frequency, or if ``X`` is invalid (not pandas,
                ``MultiIndex``, non-date or duplicated index, fewer than two
                observations, undetectable or multiplied frequency).
        """
        # Validation des paramètres
        _validate_integer(self.n_obs, 'n_obs', allow_negative=False)
        if self.how not in ('first', 'last'):
            raise ValueError(f"how must be 'first' or 'last', got {self.how!r}")

        # Validation de la structure temporelle des données
        data, _ = _validate_time_series_input(X, 'MaskTransformer')

        # Détection de la fréquence de l'index
        base, position, suffix = _detect_index_components(data.index)

        # Vérification que la fréquence de l'index est strictement supérieure à la fréquence du masque
        if not is_higher_frequency(base, self.mask_frequency):
            raise ValueError(
                "The index frequency should be strictly higher than the mask frequency. "
                f"The index frequency is {base} and the mask frequency is {self.mask_frequency}"
            )

        # Stockage des composantes de l'index et réinitialisation du stock de cellules masquées
        self.index_frequency_, self.index_position_, self.index_suffix_ = base, position, suffix
        self.index_offset_ = _index_offset(base, position, suffix)
        self.masked_values_: Dict[Any, pd.Series] = {}
        self.original_dtypes_: Dict[Any, Any] = {}

        return self

    # Méthode de transformation des données
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Set to NaN the observations on the masked positions of each period.

        Args:
            X: Time series or DataFrame indexed by dates (a single observation is
                enough).

        Returns:
            Masked data of the same type, sorted by date (masked integer columns
            become float).

        Raises:
            NotFittedError: If the transformer is not fitted.
            ValueError: If ``X`` is invalid (see ``fit``).
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self, 'index_offset_')

        # Validation de la structure temporelle des données
        data, period_freq = _validate_time_series_input(X, 'MaskTransformer')
        frame = data.to_frame(_SERIES_COLUMN) if isinstance(data, pd.Series) else data

        # Lignes situées sur une position masquée de leur période
        rows = self._rows_to_mask(frame.index)

        # Stockage des valeurs d'origine des cellules masquées
        self._store(frame, rows)

        # Masquage des lignes concernées (copie indépendante)
        condition = pd.DataFrame(np.repeat(rows[:, None], frame.shape[1], axis=1), index=frame.index, columns=frame.columns)
        masked = frame.mask(condition) if rows.any() else frame.copy()

        # Retour au type d'entrée
        if isinstance(data, pd.Series):
            masked = masked[_SERIES_COLUMN].rename(data.name)
        return _restore_index_type(masked, period_freq)

    # Méthode de transformation inverse des données
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Put back the original values of the masked cells present in ``X``.

        A cell masked by a ``transform`` call since ``fit`` recovers its original
        value, even when ``X`` carries another value there (a prediction): the
        inversion undoes the masking. Every other cell is returned as given, and
        the output has the index of ``X`` (sorted by date): stored cells whose date
        or column is not in ``X`` are ignored. Integer columns recover their dtype
        when no NaN is left.

        Args:
            X: Masked series or DataFrame (or a part of it).

        Returns:
            Data with the masked cells restored.

        Raises:
            NotFittedError: If the transformer is not fitted.
            ValueError: If ``X`` is invalid (see ``fit``).
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self, 'index_offset_')

        # Validation de la structure temporelle des données
        data, period_freq = _validate_time_series_input(X, 'MaskTransformer')
        frame = data.to_frame(_SERIES_COLUMN) if isinstance(data, pd.Series) else data.copy()

        # Restitution colonne par colonne des cellules masquées présentes dans X
        for column in frame.columns:
            stored = self.masked_values_.get(column)
            if stored is None:
                continue
            is_restored = frame.index.isin(stored.index)
            if is_restored.any():
                frame[column] = frame[column].where(~is_restored, stored.reindex(frame.index))
            # Restitution du type entier d'origine quand aucune valeur manquante ne subsiste
            frame[column] = self._restore_dtype(frame[column], self.original_dtypes_.get(column))

        # Retour au type d'entrée
        restored = frame[_SERIES_COLUMN].rename(data.name) if isinstance(data, pd.Series) else frame
        return _restore_index_type(restored, period_freq)

    # Méthode auxiliaire de repérage des lignes à masquer
    def _rows_to_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Flag the dates on the first / last ``n_obs`` positions of their period.

        Args:
            index: Sorted dates of the data.

        Returns:
            Boolean array, True for the dates to mask.
        """
        # Initialisation du masque
        n_dates = len(index)
        rows = np.zeros(n_dates, dtype=bool)
        if self.n_obs == 0 or n_dates == 0:
            return rows

        # Parcours des périodes contenant au moins une observation
        start_pos = 0
        while start_pos < n_dates:
            # Bornes calendaires [début, fin) de la période de la première date restante
            period_start, period_end = get_period_boundaries(index[start_pos], self.mask_frequency)
            end_pos = max(int(index.searchsorted(period_end, side='left')), start_pos + 1)

            # Grille régulière des positions de la période, à la fréquence de l'index
            grid = pd.date_range(start=period_start, end=period_end, freq=self.index_offset_, inclusive='left')
            if len(grid) > 0:
                # Position de chaque observation : dernier point de grille qui la précède ou l'égale
                positions = np.clip(grid.searchsorted(index[start_pos:end_pos], side='right') - 1, 0, None)
                if self.how == 'first':
                    rows[start_pos:end_pos] = positions < self.n_obs
                else:
                    rows[start_pos:end_pos] = (len(grid) - 1 - positions) < self.n_obs

            start_pos = end_pos

        return rows

    # Méthode auxiliaire de stockage des cellules masquées
    def _store(self, frame: pd.DataFrame, rows: np.ndarray) -> None:
        """Add the original values of the masked rows to the store.

        Args:
            frame: Validated data before masking.
            rows: Boolean array of the rows masked.
        """
        masked_rows = frame.loc[rows]
        for column in frame.columns:
            # Type d'origine de la colonne (pour la restitution)
            self.original_dtypes_[column] = frame[column].dtype
            if masked_rows.empty:
                continue
            new_values = masked_rows[column]
            previous = self.masked_values_.get(column)
            if previous is None:
                self.masked_values_[column] = new_values
            else:
                # Cumul des transform successifs : la dernière valeur l'emporte pour une même date
                kept = previous[~previous.index.isin(new_values.index)]
                self.masked_values_[column] = pd.concat([kept, new_values]).sort_index()

    # Méthode auxiliaire de restitution du type entier d'une colonne
    @staticmethod
    def _restore_dtype(column: pd.Series, dtype: Any) -> pd.Series:
        """Give an integer or boolean column back its dtype when it is lossless.

        Args:
            column: Restored column.
            dtype: Dtype of the column at ``transform`` time (None if unknown).

        Returns:
            The column, cast back to ``dtype`` when it has no NaN and the cast
            does not change any value; unchanged otherwise.
        """
        if dtype is None or column.dtype == dtype:
            return column
        if not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            return column
        if column.isna().any():
            return column
        try:
            converted = column.astype(dtype)
        except (TypeError, ValueError):
            return column
        return converted if (converted == column).all() else column
