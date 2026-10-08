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
from sklearn.exceptions import NotFittedError
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
from ..panel.utils import split_variable_key
from tsforecast.utils.validation import validate_temporal_data
from tsforecast.utils.parse import build_frequency_string
from tsforecast.utils._constants import BUSINESS_DAYS_PER_WEEK, DAYS_PER_WEEK
from .report import DelayFitReport, ColumnDelayRecord

# Journalisation : aucun handler n'est configuré ici, c'est à l'application d'en fournir un
logger = logging.getLogger(__name__)

# Classe d'application des délais de publication
class PublicationDelayTransformer(BaseEstimator, TransformerMixin):
    """Apply publication delays to time series or panel data.

    At ``fit``, the delay of each variable is turned into a number of periods,
    given the prediction date: with the ``'shift'`` strategy, the values are
    moved to the dates at which they are published (``n_periods`` periods of the
    detected frequency of the column, negative: later dates); with the
    ``'mask'`` strategy, the last ``n_obs`` observations of every period of the
    target frequency are hidden. ``transform`` applies these settings through
    :class:`ShiftTransformer` / :class:`MaskTransformer` (wrapped in a
    :class:`~tsforecast.panel.PanelwiseTransformer` for a panel) and
    ``inverse_transform`` reverses them.

    The number of periods of a delay ``d`` is ``-ceil((d - e) / p)``, with ``e``
    the time elapsed between the start of the period of the prediction date and
    the prediction date (minus one period when the delay is counted from the
    period end) and ``p`` the length of a period, both in the unit of the delay
    (one month = 30 days, one quarter = 91 days, one year = 365 days).

    The shift moves dates without losing values: the output index is the union
    of the shifted dates of the columns (it grows, and dates no column occupies
    any more disappear). ``inverse_transform`` restores the dates and drops the
    rows that ``transform`` added and that are empty once inverted.

    Delays specification (``delays``):

    - a dictionary ``{column: delay}`` (the unit and the reference point are then
      given by ``delay_unit`` / ``reference_point`` or ``default_values``);
    - a DataFrame with one row per variable and the column ``delay``, the
      variable being either in a column ``column`` or in the index (level
      ``'column'``, or the last level), as returned by
      ``calculate_applicable_delay``. The optional columns ``unit``,
      ``reference_point`` and ``frequency`` (target frequency of the mask)
      provide the parameters of each variable. A variable listed several times
      (per-entity delays) is rejected: use :func:`create_delay_transformer_factory`
      with a :class:`~tsforecast.panel.PanelwiseTransformer`.

    A ``NaN`` delay is an unknown delay, handled as a missing one.

    Parameters are resolved per variable, an explicit argument winning over the
    delays table, itself winning over ``default_values``.

    Args:
        delays: Delays specification (see above).
        prediction_date: Prediction date (anything ``resolve_date`` accepts, 'today'
            by default).
        strategy: ``'shift'``, ``'mask'``, or a ``{column: strategy}`` dictionary.
            Columns with a delay but absent from the dictionary are left unchanged
            (with a warning), and ``default_values`` is ignored.
        target_frequency: Frequency of the masked periods ('mask' strategy), for all
            columns or per column. Ignored by the shift.
        delay_unit: Unit of the delays ('D', 'h', 'W', 'day', ...), for all columns or
            per column.
        reference_point: Origin of the delays, ``'start'`` or ``'end'`` of the period,
            for all columns or per column.
        handle_missing_delays: Columns of ``X`` without a known delay are left
            unchanged; ``'warn'`` (default) announces them, ``'error'`` rejects them,
            ``'ignore'`` stays silent.
        default_values: Values used for the columns of ``X`` lacking one, with the
            keys ``'delay'``, ``'unit'``, ``'reference_point'`` (and
            ``'target_frequency'`` for the 'mask' strategy). With it, every column of
            ``X`` gets a delay. Ignored when ``strategy`` is a dictionary.

    Attributes:
        prediction_date_: Resolved prediction date.
        inferred_params_: Parameters read from the delays table: ``{'delay_unit': ...,
            'reference_point': ..., 'target_frequency': ...}``, each a
            ``{column: value}`` dictionary.
        detected_frequencies_: Detected frequency (base code) of each column of ``X``,
            ``None`` when undetectable (for a panel, the frequency shared by the
            entities).
        shift_params: ``{column: {'n_periods': int, 'frequency': str}}`` of the
            shifted columns.
        mask_params: ``{column: {'n_obs': int, 'mask_frequency': str, 'how': 'last'}}``
            of the masked columns.
        fit_report_: :class:`~tsforecast.delays.DelayFitReport` of the last ``fit``:
            resolved setting of each column and its origin, columns ignored or
            unaffected, defaults imputed, mask-to-shift fallbacks.
        auxiliary_transformers_: Helpers fitted by the last ``transform``, used by
            ``inverse_transform`` (absent before the first ``transform``).

    Raises:
        ValueError: At construction, for an invalid ``strategy``, ``reference_point``,
            ``handle_missing_delays`` or an incomplete ``default_values``.
        TypeError: At construction, for a ``strategy`` neither string nor dictionary.

    Warns:
        UserWarning: For panel data (same delays for every entity), columns without
            delay (``handle_missing_delays='warn'``), defaults imputed, delayed columns
            without detectable frequency, impossible masks moved to the shift.

    Examples:
        >>> import pandas as pd
        >>> index = pd.date_range('2023-01-01', periods=12, freq='MS')
        >>> X = pd.DataFrame({'GDP': range(12), 'CPI': range(12)}, index=index, dtype=float)
        >>> delays = pd.DataFrame({
        ...     'column': ['GDP', 'CPI'],
        ...     'delay': [45.0, 20.0],
        ...     'unit': ['D', 'D'],
        ...     'reference_point': ['start', 'start'],
        ...     'frequency': ['M', 'M'],
        ... })
        >>> transformer = PublicationDelayTransformer(delays=delays, prediction_date='2023-12-15')
        >>> shifted = transformer.fit_transform(X)
        >>> transformer.shift_params['GDP']
        {'n_periods': -2, 'frequency': 'M'}
        >>> float(shifted.loc['2023-12-01', 'GDP'])  # value of October, published by December 15
        9.0
        >>> transformer.inverse_transform(shifted).equals(X)
        True
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
            delays: Delays specification: ``{column: delay}`` or a delays table.
            prediction_date: Prediction date.
            strategy: 'shift', 'mask' or a ``{column: strategy}`` dictionary.
            target_frequency: Target frequency of the mask, for all or per column.
            delay_unit: Unit of the delays, for all or per column.
            reference_point: 'start' or 'end', for all or per column.
            handle_missing_delays: 'ignore', 'warn' or 'error'.
            default_values: Defaults for the columns lacking a value.

        Raises:
            ValueError: If a parameter has an invalid value.
            TypeError: If ``strategy`` is neither a string nor a dictionary.
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

        # Paramètre de point de référence : valeur unique ou dictionnaire par variable
        if isinstance(reference_point, dict):
            for k, v in reference_point.items():
                if v not in ['start', 'end']:
                    raise ValueError(f"reference_point must be 'start' or 'end', for variable '{k}' got '{v}'")
        elif reference_point is not None and reference_point not in ['start', 'end']:
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
        """Resolve the delay settings of every column at the prediction date.

        Args:
            X: Time series (``DataFrame`` or named ``Series`` indexed by dates) or panel
                data (``MultiIndex``: entities on the first levels, dates on the last
                one). For a panel, the same delays apply to every entity.
            y: Ignored.

        Returns:
            self: The fitted transformer.

        Raises:
            ValueError: If the delays table is invalid, if a delayed column has no
                resolved unit, reference point or (mask) target frequency, if
                ``handle_missing_delays='error'`` and a column has no delay, if no
                column of ``X`` has a detectable frequency, or, for a panel, if a
                delayed column (or the index, for the mask) has different
                frequencies across entities.
        """
        # Conversion d'une Series en DataFrame et détection de la structure de panel
        X, _ = _as_frame(X)
        is_panel = is_panel_data(X)
        if is_panel:
            # Warning
            warnings.warn(
                "Panel data: the same publication delays are applied to every entity. Use "
                "create_delay_transformer_factory (or prepare_entity_kwargs_from_delays) with a "
                "PanelwiseTransformer to apply per-entity delays."
            )

        # Oubli des transformateurs auxiliaires d'un transform précédent (paramètres périmés)
        for attribute in ('auxiliary_transformers_', 'transform_input_index_'):
            if hasattr(self, attribute):
                delattr(self, attribute)

        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(self.prediction_date)

        # Tableau des délais à plat (variable dans la colonne 'column') et paramètres qui s'en déduisent
        delays_table = self._delays_table()
        self.inferred_params_ = self._infer_parameters_from_delays(delays_table)

        # Délais connus de chaque variable, complétés par le délai par défaut
        specified_delays = (dict(zip(delays_table['column'], delays_table['delay']))
                            if delays_table is not None else dict(self.delays))
        delays_dict, default_delay_columns = self._build_delays_dict(X, specified_delays)

        # Répartition des variables retardées entre décalage et masquage
        shift_columns, mask_columns = self._split_columns_by_strategy(X, delays_dict)

        # Variables de X sans délai connu
        self._handle_missing_delays([col for col in X.columns if col not in delays_dict])

        # Résolution des paramètres des variables retardées (explicite > inféré > défaut)
        delayed_columns = shift_columns + mask_columns
        delay_unit_dict, delay_unit_sources = self._resolve_parameter(
            columns=delayed_columns, param_name='delay_unit', explicit_value=self.delay_unit,
            inferred_key='delay_unit', default_key='unit')
        reference_point_dict, reference_point_sources = self._resolve_parameter(
            columns=delayed_columns, param_name='reference_point', explicit_value=self.reference_point,
            inferred_key='reference_point', default_key='reference_point')
        target_frequency_dict, target_frequency_sources = self._resolve_parameter(
            columns=mask_columns, param_name='target_frequency', explicit_value=self.target_frequency,
            inferred_key='target_frequency', default_key='target_frequency')
        self._check_resolved(delayed_columns, 'delay_unit', delay_unit_dict)
        self._check_resolved(delayed_columns, 'reference_point', reference_point_dict)
        self._check_resolved(mask_columns, 'target_frequency', target_frequency_dict)

        # Détection des fréquences par colonne (fréquence commune aux entités pour un panel)
        self.detected_frequencies_ = self._detect_column_frequencies(X, delayed_columns, is_panel)

        # Les variables sans fréquence détectable sont laissées telles quelles
        undetected = [col for col in delayed_columns if self.detected_frequencies_[col] is None]
        if undetected:
            # Warning
            warnings.warn(
                f"Could not detect the frequency of the delayed columns {undetected} (all NaN or too few "
                f"observations): they are left unchanged"
            )
            shift_columns = [col for col in shift_columns if col not in undetected]
            mask_columns = [col for col in mask_columns if col not in undetected]

        # Calcul du nombre de périodes à shifter pour chaque variable
        self.shift_params = {}
        for col in shift_columns:
            n_periods = self._compute_shift_periods(
                col=col,
                delays_dict=delays_dict,
                delay_unit_dict=delay_unit_dict,
                reference_point_dict=reference_point_dict
            )
            self.shift_params[col] = {'n_periods': n_periods, 'frequency': self.detected_frequencies_[col]}

        # Calcul du nombre d'observations à masquer pour chaque variable
        self.mask_params = {}
        mask_fallbacks: List[str] = []
        index_frequency = self._index_frequency(X) if mask_columns else None
        for col in mask_columns:
            result = self._compute_mask_periods(
                col=col,
                index_frequency=index_frequency,
                delays_dict=delays_dict,
                delay_unit_dict=delay_unit_dict,
                reference_point_dict=reference_point_dict,
                target_frequency_dict=target_frequency_dict
            )
            # Distinction suivant que le masquage est possible ou non
            if result['can_mask']:
                # Un nombre négatif signifie une donnée déjà publiée : rien à masquer
                self.mask_params[col] = {
                    'n_obs': max(0, result['n_periods']),
                    'mask_frequency': result['target_frequency'],
                    'how': 'last'
                }
            else:
                # Warning
                warnings.warn(f"Could not mask the column '{col}' because it would have created a series of Nan. Moved it to the shifted columns")
                # Repli sur le décalage de la stratégie 'shift' (périodes de la colonne, vers les dates ultérieures)
                n_periods = self._compute_shift_periods(
                    col=col,
                    delays_dict=delays_dict,
                    delay_unit_dict=delay_unit_dict,
                    reference_point_dict=reference_point_dict
                )
                self.shift_params[col] = {'n_periods': n_periods, 'frequency': self.detected_frequencies_[col]}
                mask_fallbacks.append(col)

        # Rapport d'ajustement : tout ce que le fit a résolu, sans avertissement à relire
        self.fit_report_ = self._build_fit_report(
            X=X,
            specified_delays=specified_delays,
            delays_dict=delays_dict,
            delay_unit_dict=delay_unit_dict,
            reference_point_dict=reference_point_dict,
            sources={
                'delay': {col: 'default' if col in default_delay_columns else 'explicit' for col in delays_dict},
                'delay_unit': delay_unit_sources,
                'reference_point': reference_point_sources,
                'target_frequency': target_frequency_sources,
            },
            mask_fallbacks=mask_fallbacks
        )
        # Logging
        logger.info(self.fit_report_.summary())

        return self

    # Méthode auxiliaire de mise à plat du tableau des délais
    def _delays_table(self) -> Optional[pd.DataFrame]:
        """Return the delays table with the variable in a ``column`` column.

        Returns:
            The flat table, or None when ``delays`` is a dictionary.

        Raises:
            ValueError: If the table has no ``delay`` column or lists a variable twice.
        """
        # Spécification sous forme de dictionnaire
        if not isinstance(self.delays, pd.DataFrame):
            return None
        table = self.delays
        if 'delay' not in table.columns:
            raise ValueError("The delays table must have a 'delay' column")

        # Variable dans l'index : niveau 'column', à défaut le dernier niveau (convention des fabriques)
        if 'column' not in table.columns:
            level = 'column' if 'column' in table.index.names else table.index.nlevels - 1
            variables = table.index.get_level_values(level)
            table = table.reset_index(drop=True).assign(column=np.asarray(variables, dtype=object))

        # Un délai par variable : les délais par entité relèvent des fabriques
        duplicated = table.loc[table['column'].duplicated(), 'column'].unique().tolist()
        if duplicated:
            raise ValueError(
                f"The delays table lists the variables {duplicated} several times. For per-entity delays, use "
                f"create_delay_transformer_factory (or prepare_entity_kwargs_from_delays) with a PanelwiseTransformer."
            )
        return table

    # Méthode auxiliaire d'inférence des paramètres d'unité du délai, de point de référence et de fréquence cible
    def _infer_parameters_from_delays(self, delays_table: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Read delay_unit, reference_point and target_frequency from the delays table.

        Args:
            delays_table: Flat delays table, or None.

        Returns:
            Dict with the keys 'delay_unit', 'reference_point' and 'target_frequency',
            each mapping the variables to their value (missing values left out).
        """
        # Association entre colonne du tableau et paramètre inféré
        sources = {'delay_unit': 'unit', 'reference_point': 'reference_point', 'target_frequency': 'frequency'}
        inferred: Dict[str, Dict[str, Any]] = {key: {} for key in sources}
        if delays_table is None:
            return inferred
        for key, column in sources.items():
            if column in delays_table.columns:
                inferred[key] = {var: value for var, value in zip(delays_table['column'], delays_table[column])
                                 if not _is_missing(value)}
        return inferred

    # Méthode auxiliaire de construction du dictionnaire des délais connus
    def _build_delays_dict(
        self,
        X: pd.DataFrame,
        specified_delays: Dict[Any, Any]
    ) -> Tuple[Dict[Any, float], List[Any]]:
        """Keep the known delays and give ``default_values['delay']`` to the columns of ``X`` without one.

        Args:
            X: Data to fit on.
            specified_delays: Delays of the specification (``NaN`` = unknown).

        Returns:
            Tuple ``(delays_dict, default_delay_columns)``.
        """
        # Délais connus (un délai NaN est un délai inconnu)
        delays_dict = {col: delay for col, delay in specified_delays.items() if not _is_missing(delay)}

        # Délai par défaut des colonnes de X sans délai connu (stratégie unique seulement)
        default_delay_columns = []
        if self.default_values is not None and isinstance(self.strategy, str):
            for col in X.columns:
                if col not in delays_dict:
                    delays_dict[col] = self.default_values['delay']
                    default_delay_columns.append(col)
                    warnings.warn(f"Imputed default delay value '{self.default_values['delay']}' for column '{col}'")
        return delays_dict, default_delay_columns

    # Méthode auxiliaire de répartition des variables entre décalage et masquage
    def _split_columns_by_strategy(self, X: pd.DataFrame, delays_dict: Dict[Any, float]) -> Tuple[List[Any], List[Any]]:
        """Split the delayed columns of ``X`` between the shift and the mask, in the order of ``X``.

        Args:
            X: Data to fit on.
            delays_dict: Known delays.

        Returns:
            Tuple ``(shift_columns, mask_columns)``.
        """
        # Colonnes auxquelles appliquer des délais
        delayed = [col for col in X.columns if col in delays_dict]
        # Stratégie unique
        if isinstance(self.strategy, str):
            return (delayed, []) if self.strategy == 'shift' else ([], delayed)

        # Stratégie par variable : les variables retardées absentes du dictionnaire sont laissées telles quelles
        without_strategy = [col for col in delayed if col not in self.strategy]
        if without_strategy:
            # Warning
            warnings.warn(
                f"The columns {without_strategy} have a delay but no strategy in the 'strategy' dictionary: "
                f"they are left unchanged"
            )
        return ([col for col in delayed if self.strategy.get(col) == 'shift'],
                [col for col in delayed if self.strategy.get(col) == 'mask'])

    # Méthode auxiliaire de traitement des variables sans délai
    def _handle_missing_delays(self, columns: List[Any]) -> None:
        """Apply ``handle_missing_delays`` to the columns of ``X`` without a known delay.

        Args:
            columns: Columns without delay.

        Raises:
            ValueError: If ``handle_missing_delays='error'`` and ``columns`` is not empty.
        """
        # Ne fait rien si aucune colonne n'est spécifiée
        if not columns:
            return
        # Construction du message
        message = f"No publication delay for the columns {columns}: they are left unchanged"
        if self.handle_missing_delays == 'error':
            raise ValueError(message + " (handle_missing_delays='error')")
        if self.handle_missing_delays == 'warn':
            warnings.warn(message)

    # Méthode auxiliaire de résolution d'un paramètre
    def _resolve_parameter(
        self,
        columns: List[Any],
        param_name: str,
        explicit_value: Optional[Union[str, Dict[str, str]]],
        inferred_key: str,
        default_key: str
    ) -> Tuple[Dict[Any, Any], Dict[Any, str]]:
        """Resolve a parameter for each column: explicit > inferred > default.

        Args:
            columns: Columns needing the parameter.
            param_name: Name of the parameter (warning messages).
            explicit_value: Value given to the constructor (str, dict or None).
            inferred_key: Key of the parameter in ``inferred_params_``.
            default_key: Key of the parameter in ``default_values``.

        Returns:
            Tuple ``(values, sources)``: value and origin ('explicit', 'inferred' or
            'default') of each resolved column; unresolved columns are left out.
        """
        # Initialisation des dictionnaires des valeurs à la source et de délai applicables
        values: Dict[Any, Any] = {}
        sources: Dict[Any, str] = {}
        inferred = self.inferred_params_.get(inferred_key, {})
        # Valeurs par défaut utilisables (ignorées avec une stratégie par variable)
        use_default = (self.default_values is not None and isinstance(self.strategy, str)
                       and default_key in self.default_values)
        # Parcours des colonnes
        for col in columns:
            if isinstance(explicit_value, str) or (isinstance(explicit_value, dict) and col in explicit_value):
                values[col] = explicit_value if isinstance(explicit_value, str) else explicit_value[col]
                sources[col] = 'explicit'
            elif col in inferred:
                values[col] = inferred[col]
                sources[col] = 'inferred'
            elif use_default:
                values[col] = self.default_values[default_key]
                sources[col] = 'default'
                warnings.warn(f"Imputed default {param_name} value '{values[col]}' for column '{col}'")
        return values, sources

    # Méthode auxiliaire de vérification des paramètres résolus
    @staticmethod
    def _check_resolved(columns: List[Any], param_name: str, resolved: Dict[Any, Any]) -> None:
        """Reject the delayed columns whose parameter could not be resolved.

        Args:
            columns: Columns needing the parameter.
            param_name: Name of the parameter.
            resolved: Resolved values.

        Raises:
            ValueError: If some columns have no value, naming them.
        """
        unresolved = [col for col in columns if col not in resolved]
        if unresolved:
            raise ValueError(
                f"No '{param_name}' for the delayed columns {unresolved}: give it to the constructor, in the "
                f"delays table, or in 'default_values'"
            )

    # Méthode auxiliaire de détection des fréquences par colonne
    def _detect_column_frequencies(
        self,
        X: pd.DataFrame,
        delayed_columns: List[Any],
        is_panel: bool
    ) -> Dict[Any, Optional[str]]:
        """Detect the frequency (base code) of each column of ``X``.

        For a panel, the frequency of a column is the one its entities share.

        Args:
            X: Data to fit on.
            delayed_columns: Delayed columns (their frequency must be shared by the entities).
            is_panel: Whether ``X`` is panel data.

        Returns:
            ``{column: frequency or None}``.

        Raises:
            ValueError: If no column has a detectable frequency, or if a delayed column of a
                panel has different frequencies across entities.
        """
        # Détecttion des fréquences
        detected = detect_frequency(data=X, time_col=None, panel_cols=None, check_consistency=False, strict=False)
        if not is_panel:
            frequencies = {col: detected.get(col) for col in X.columns}
        else:
            # Fréquences de chaque colonne sur les entités où elle est détectable
            per_column: Dict[Any, set] = {col: set() for col in X.columns}
            for key, frequency in detected.items():
                _, col = split_variable_key(key)
                if frequency is not None:
                    per_column[col].add(frequency)
            conflicts = {col: sorted(found) for col, found in per_column.items()
                         if col in delayed_columns and len(found) > 1}
            if conflicts:
                raise ValueError(
                    f"The columns {list(conflicts)} have different frequencies across entities ({conflicts}): use "
                    f"create_delay_transformer_factory with a PanelwiseTransformer to apply per-entity delays."
                )
            frequencies = {col: next(iter(found)) if found else None for col, found in per_column.items()}

        # Aucune fréquence détectable alors que des colonnes sont retardées : jeu vide, d'une seule observation ou irrégulier
        if delayed_columns and all(frequency is None for frequency in frequencies.values()):
            raise ValueError(
                "Could not detect the frequency of any column of X: at least two observations on a "
                "regular grid are needed"
            )
        return frequencies

    # Méthode auxiliaire de détection de la fréquence de l'index
    @staticmethod
    def _index_frequency(X: pd.DataFrame) -> str:
        """Detect the frequency of the index (shared by the entities for a panel).

        Args:
            X: Data to fit on.

        Returns:
            Base code of the index frequency.

        Raises:
            ValueError: If the entities of a panel have different index frequencies.
        """
        # Détection de la fréquence de l'index
        detected = detect_index_frequency(X.index)
        if not isinstance(detected, dict):
            return detected
        # Tri et unicisation des fréquences renseignées
        found = sorted({frequency for frequency in detected.values() if frequency is not None})
        if len(found) > 1:
            raise ValueError(
                f"The entities have different index frequencies ({found}): use create_delay_transformer_factory "
                f"with a PanelwiseTransformer to mask per entity."
            )
        return found[0] if found else None

    # Méthode auxiliaire de construction du rapport d'ajustement
    def _build_fit_report(
        self,
        X: pd.DataFrame,
        specified_delays: Dict[Any, Any],
        delays_dict: Dict[Any, float],
        delay_unit_dict: Dict[Any, str],
        reference_point_dict: Dict[Any, str],
        sources: Dict[str, Dict[Any, str]],
        mask_fallbacks: List[Any]
    ) -> DelayFitReport:
        """Build the :class:`DelayFitReport` of the fit from the resolved parameters.

        Args:
            X: Data the transformer was fitted on.
            specified_delays: Delays of the specification.
            delays_dict: Known (or default) delay of each variable.
            delay_unit_dict: Resolved delay unit of each variable.
            reference_point_dict: Resolved reference point of each variable.
            sources: Origin of each parameter (``'delay'``, ``'delay_unit'``,
                ``'reference_point'``, ``'target_frequency'``) per variable.
            mask_fallbacks: Variables moved from mask to shift.

        Returns:
            The immutable fit report.
        """
        # Une ligne par variable retardée, 'shift' d'abord puis 'mask' (ordre de l'application)
        records = []
        # Parcours des paramètres de 'shift' et de 'mask'
        for strategy, params_dict in (('shift', self.shift_params), ('mask', self.mask_params)):
            # Parcours des colonnes et des apramètres associés à la stratégie
            for col, params in params_dict.items():
                is_mask = strategy == 'mask'
                records.append(ColumnDelayRecord(
                    column=col,
                    strategy=strategy,
                    delay=delays_dict.get(col),
                    delay_unit=delay_unit_dict.get(col),
                    reference_point=reference_point_dict.get(col),
                    frequency=self.detected_frequencies_.get(col),
                    n_periods=None if is_mask else params['n_periods'],
                    n_obs=params['n_obs'] if is_mask else None,
                    target_frequency=params['mask_frequency'] if is_mask else None,
                    delay_unit_source=sources['delay_unit'].get(col),
                    reference_point_source=sources['reference_point'].get(col),
                    target_frequency_source=sources['target_frequency'].get(col) if is_mask else None,
                    moved_from_mask=(not is_mask) and (col in mask_fallbacks),
                    delay_source=sources['delay'].get(col)
                ))

        # Couples (variable, paramètre) complétés par les valeurs par défaut
        defaults_imputed = tuple(
            (record.column, name)
            for record in records
            for name in ('delay', 'delay_unit', 'reference_point', 'target_frequency')
            if getattr(record, f'{name}_source') == 'default'
        )

        # Variables de X sans délai appliqué, et variables de la spécification des délais absentes de X
        delayed = {record.column for record in records}
        return DelayFitReport(
            prediction_date=self.prediction_date_,
            columns=tuple(records),
            columns_unaffected=tuple(col for col in X.columns if col not in delayed),
            columns_ignored=tuple(col for col in specified_delays if col not in X.columns),
            defaults_imputed=defaults_imputed,
            mask_fallbacks=tuple(mask_fallbacks)
        )

    # Méthode de transformation des données
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply the publication delays to the data.

        Args:
            X: Data with the structure seen at ``fit`` (time series or panel).

        Returns:
            Data of the same type with the delays applied: shifted columns moved to
            their publication dates (index = union of the dates of the columns),
            masked cells set to ``NaN``, other columns unchanged, columns in the input
            order.

        Raises:
            NotFittedError: If the transformer is not fitted.
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self, 'shift_params')

        # Conversion d'une Series en DataFrame et détection de la structure de panel
        X, series_name = _as_frame(X)
        is_panel = is_panel_data(X)

        # Transformateurs auxiliaires et index d'entrée, conservés pour l'inversion
        self.auxiliary_transformers_: Dict[str, Dict[tuple, BaseEstimator]] = {'shift': {}, 'mask': {}}
        self.transform_input_index_ = X.index

        # Traitement des variables à décaler puis à masquer
        list_df_transformed = self._apply_auxiliary_transformers(
            X=X, params_dict=self.shift_params, transformer_class=ShiftTransformer,
            transformer_type='shift', is_panel=is_panel)
        list_df_transformed.extend(self._apply_auxiliary_transformers(
            X=X, params_dict=self.mask_params, transformer_class=MaskTransformer,
            transformer_type='mask', is_panel=is_panel))

        return _restore_series(self._assemble(list_df_transformed, X), series_name)

    # Méthode de transformation inverse des données
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse the publication delays applied by the last ``transform``.

        The rows added by ``transform`` (dates outside its input) that are empty once
        inverted are dropped, so that a round trip restores the input index.

        Args:
            X: Transformed data.

        Returns:
            Data of the same type with the delays reversed.

        Raises:
            NotFittedError: If the transformer is not fitted, or if ``transform`` was
                not called since the last ``fit`` (the inversion uses its helpers).
        """
        # Vérification que le transformer est entraîné, puis qu'un transform a fourni ses auxiliaires
        check_is_fitted(self, 'shift_params')
        if not hasattr(self, 'auxiliary_transformers_'):
            raise NotFittedError(
                "inverse_transform reverses the last transform: call transform before inverse_transform"
            )

        # Conversion d'une Series en DataFrame
        X, series_name = _as_frame(X)

        # Inversion des décalages puis des masques
        list_df_inversed = self._apply_inverse_transformers(X=X, params_dict=self.shift_params, transformer_type='shift')
        list_df_inversed.extend(self._apply_inverse_transformers(X=X, params_dict=self.mask_params, transformer_type='mask'))
        df_inversed = self._assemble(list_df_inversed, X)

        # Suppression des lignes ajoutées par transform, vides une fois inversées
        added = ~df_inversed.index.isin(self.transform_input_index_) & df_inversed.isna().all(axis=1).to_numpy()
        return _restore_series(df_inversed[~added], series_name)

    # Méthode auxiliaire d'assemblage des colonnes transformées et non transformées
    @staticmethod
    def _assemble(parts: List[pd.DataFrame], X: pd.DataFrame) -> pd.DataFrame:
        """Join the transformed column groups and the untouched columns, in the order of ``X``.

        Args:
            parts: Transformed column groups.
            X: Input data.

        Returns:
            The assembled frame (outer join on the index).
        """
        # Jointure sur l'index des données transformées (aucune colonne transformée : index seul)
        assembled = pd.concat(parts, axis=1, join='outer') if parts else pd.DataFrame(index=X.index)
        # Ajout des colonnes non transformées
        untouched = [col for col in X.columns if col not in assembled.columns]
        if untouched:
            assembled = pd.concat([assembled, X[untouched]], axis=1, join='outer')
        # Restauration de l'ordre original des colonnes
        return assembled[X.columns]

    # Méthode auxiliaire de calcul du nombre de périodes à shifter
    def _compute_shift_periods(
        self,
        col: str,
        delays_dict: Dict[str, float],
        delay_unit_dict: Dict[str, str],
        reference_point_dict: Dict[str, str]
    ) -> int:
        """Compute the number of periods to shift for a given column.

        Args:
            col: Column name to compute shift periods for.
            delays_dict: Dictionary mapping column names to delay values.
            delay_unit_dict: Dictionary mapping column names to delay units.
            reference_point_dict: Dictionary mapping column names to reference points.

        Returns:
            Number of periods of the column frequency to shift (negative: later dates).
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
        index_frequency: str,
        delays_dict: Dict[str, float],
        delay_unit_dict: Dict[str, str],
        reference_point_dict: Dict[str, str],
        target_frequency_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """Compute the number of observations to mask for a given column.

        Args:
            col: Column name to compute mask periods for.
            index_frequency: Frequency of the index of ``X``.
            delays_dict: Dictionary mapping column names to delay values.
            delay_unit_dict: Dictionary mapping column names to delay units.
            reference_point_dict: Dictionary mapping column names to reference points.
            target_frequency_dict: Dictionary mapping column names to target frequencies.

        Returns:
            Dictionary with keys:
                - 'n_periods': Number of index observations to mask.
                - 'target_frequency': Normalized target frequency.
                - 'can_mask': Whether masking leaves at least one observation per target period.
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

        Columns sharing the same parameters go through a single helper, wrapped in a
        PanelwiseTransformer for panel data.

        Args:
            X: Input DataFrame.
            params_dict: Dictionary mapping column names to transformation parameters.
            transformer_class: Class of transformer to use (ShiftTransformer or MaskTransformer).
            transformer_type: Type identifier ('shift' or 'mask') for storage.
            is_panel: Whether the data is panel data (MultiIndex).

        Returns:
            List of transformed DataFrames, one per unique parameter combination.
        """
        list_df_transformed = []
        # Les masques nuls ne transforment rien : colonnes laissées telles quelles
        for params_key, columns in _group_columns_by_params(_active_params(params_dict, transformer_type)).items():
            params = dict(params_key)
            # Distinction suivant la structure de panel
            if is_panel:
                transformer_ = PanelwiseTransformer(transformer=transformer_class(**params), time_col=None,
                                                    panel_cols=None)
            else:
                transformer_ = transformer_class(**params)
            # Stockage du transformer et transformation des données
            self.auxiliary_transformers_[transformer_type][params_key] = transformer_
            list_df_transformed.append(transformer_.fit_transform(X[columns]))
        return list_df_transformed

    # Méthode auxiliaire d'application des transformations inverses
    def _apply_inverse_transformers(
        self,
        X: pd.DataFrame,
        params_dict: Dict[str, Dict],
        transformer_type: str
    ) -> List[pd.DataFrame]:
        """Apply inverse transformations using the helpers stored by ``transform``.

        Args:
            X: Transformed DataFrame to inverse.
            params_dict: Dictionary mapping column names to transformation parameters.
            transformer_type: Type identifier ('shift' or 'mask') for retrieval.

        Returns:
            List of inverse-transformed DataFrames, one per unique parameter combination.
        """
        return [
            self.auxiliary_transformers_[transformer_type][params_key].inverse_transform(X[columns])
            for params_key, columns in _group_columns_by_params(_active_params(params_dict, transformer_type)).items()
        ]


# Fonction de regroupement des colonnes de mêmes paramètres
def _group_columns_by_params(params_dict: Dict[Any, Dict]) -> Dict[tuple, List[Any]]:
    """Group the columns by identical parameters, in order of first appearance.

    Args:
        params_dict: Dictionary mapping column names to transformation parameters.

    Returns:
        ``{sorted parameter items: columns}``.

    Examples:
        >>> _group_columns_by_params({'a': {'n': 1}, 'b': {'n': 2}, 'c': {'n': 1}})
        {(('n', 1),): ['a', 'c'], (('n', 2),): ['b']}
    """
    groups: Dict[tuple, List[Any]] = {}
    for col, params in params_dict.items():
        groups.setdefault(tuple(sorted(params.items())), []).append(col)
    return groups


# Fonction de détection d'une valeur manquante
def _is_missing(value: Any) -> bool:
    """Tell whether a scalar value of a delays specification is missing (None or NaN).

    Args:
        value: Scalar value.

    Returns:
        True for None and NaN.

    Examples:
        >>> _is_missing(float('nan')), _is_missing(None), _is_missing(0.0), _is_missing('D'), _is_missing([1, 2])
        (True, True, False, False, False)
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Valeur non scalaire : jamais manquante
        return False


# Fonction de conversion d'une Series en DataFrame
def _as_frame(X: Union[pd.Series, pd.DataFrame]) -> Tuple[pd.DataFrame, Any]:
    """Return ``X`` as a DataFrame, and a marker to restore a Series.

    Args:
        X: Series or DataFrame.

    Returns:
        Tuple ``(frame, series_name)``: ``series_name`` is the name of the Series
        (wrapped in a 1-tuple, so that a ``None`` name is kept), or None for a DataFrame.

    Raises:
        TypeError: If ``X`` is neither a Series nor a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X, None
    if isinstance(X, pd.Series):
        return X.to_frame(name=X.name if X.name is not None else 0), (X.name,)
    raise TypeError(f"X must be a pandas Series or DataFrame, got {type(X).__name__}")


# Fonction de restauration d'une Series
def _restore_series(frame: pd.DataFrame, series_name: Any) -> Union[pd.Series, pd.DataFrame]:
    """Turn a one-column frame back into a Series when the input was one.

    Args:
        frame: Output frame.
        series_name: Marker returned by :func:`_as_frame`.

    Returns:
        The Series (with its original name), or ``frame`` unchanged.
    """
    if series_name is None:
        return frame
    return frame.iloc[:, 0].rename(series_name[0])

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
    """Create a factory of per-entity ``PublicationDelayTransformer`` from a delays table.

    The factory is meant to be passed as the ``transformer`` of a
    :class:`~tsforecast.panel.PanelwiseTransformer`: called with an entity key, it
    returns a transformer configured with the delays of that entity.

    Args:
        df_delays: Per-entity delays table, as returned by
            ``calculate_applicable_delay(..., aggregate_by_panel=True)``: a
            ``MultiIndex`` whose last level is the variable and whose other levels
            identify the entity (levels located by position, names free), and the
            columns of the delay, its unit, its reference point and its target
            frequency.
        strategy: Delay application strategy:

            - ``'shift'`` or ``'mask'``: for every entity;
            - a dict keyed by entity (tuple, or scalar for a single entity level):
              strategy of each entity;
            - a dict keyed by variable names: per-variable strategy for every entity;
            - a callable ``entity_key -> strategy``.
        prediction_date: Prediction date, shared by all the transformers.
        delay_col: Column of the delays. Defaults to 'delay'.
        unit_col: Column of the delay units. Defaults to 'unit'.
        reference_point_col: Column of the reference points. Defaults to 'reference_point'.
        target_frequency_col: Column of the target frequencies (used by the mask
            only). Defaults to 'frequency'.
        default_transformer_kwargs: Other ``PublicationDelayTransformer`` arguments,
            passed to every transformer (``prediction_date`` excepted).

    Returns:
        Callable taking an entity key (tuple, or scalar for a single entity level)
        and returning a new configured transformer. A parameter constant within an
        entity is passed as a scalar, a varying one as a ``{variable: value}`` dict.

    Raises:
        ValueError: If required columns are missing or if ``df_delays`` has no
            entity level.
        KeyError: When called with an entity absent from ``df_delays`` (the
            available entities are listed), or absent from a per-entity strategy dict.
        ValueError: When called, if the strategy (or the callable's result) is invalid.
        TypeError: When called, if ``strategy`` is neither a string, a dict nor a callable.

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.panel import PanelwiseTransformer
        >>> index = pd.MultiIndex.from_tuples(
        ...     [('FR', 'GDP'), ('DE', 'GDP')], names=['country', 'column'])
        >>> delays = pd.DataFrame({'delay': [45.0, 75.0], 'unit': 'D', 'frequency': 'M',
        ...                        'reference_point': 'start'}, index=index)
        >>> factory = create_delay_transformer_factory(delays, prediction_date='2023-12-15')
        >>> factory('DE').delays
        {'GDP': 75.0}
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        >>> X = pd.concat({'FR': pd.DataFrame({'GDP': range(12)}, index=dates, dtype=float),
        ...                'DE': pd.DataFrame({'GDP': range(12)}, index=dates, dtype=float)},
        ...               names=['country', 'date'])
        >>> shifted = PanelwiseTransformer(transformer=factory, time_col=None).fit_transform(X)
        >>> shifted.loc['FR', 'GDP'].last_valid_index().strftime('%Y-%m'), shifted.loc['DE', 'GDP'].last_valid_index().strftime('%Y-%m')
        ('2024-02', '2024-03')
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
            entity_key: Entity identifier (tuple, or scalar for a single entity level).

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

        # Construction des kwargs du transformer
        entity_strategy = _resolve_strategy(strategy, entity_key)
        transformer_kwargs = {**base_kwargs, **_entity_kwargs(entity_params[entity_key], entity_strategy)}

        # Création et retour du transformer
        return PublicationDelayTransformer(**transformer_kwargs)

    return transformer_factory


# Fonction de construction des kwargs d'une entité
def _entity_kwargs(params: Dict[str, Any], entity_strategy: Union[str, Dict[str, str]]) -> Dict[str, Any]:
    """Build the ``PublicationDelayTransformer`` arguments of an entity.

    The target frequency only matters to the mask: it is not passed with the
    'shift' strategy (where the transformer would announce it as ignored).

    Args:
        params: Parameters of the entity (``_build_entity_params``).
        entity_strategy: Resolved strategy of the entity.

    Returns:
        Keyword arguments ``delays``, ``delay_unit``, ``reference_point``,
        ``target_frequency`` and ``strategy``.

    Examples:
        >>> _entity_kwargs({'delays': {'GDP': 45.0}, 'delay_unit': 'D', 'reference_point': 'end',
        ...                 'target_frequency': 'Q'}, 'shift')['target_frequency'] is None
        True
    """
    return {
        'delays': params['delays'],
        'delay_unit': params['delay_unit'],
        'reference_point': params['reference_point'],
        'target_frequency': None if entity_strategy == 'shift' else params['target_frequency'],
        'strategy': entity_strategy
    }


# Fonction auxiliaire de construction des dictionnaires de paramètres pour chaque entité
def _build_entity_params(
    df_delays: pd.DataFrame,
    delay_col: str,
    unit_col: str,
    reference_point_col: str,
    target_frequency_col: str
) -> Dict[tuple, Dict[str, Any]]:
    """Build the parameters of each entity of a per-entity delays table.

    Args:
        df_delays: Delays table (entity levels, then the variable as last level).
        delay_col: Column name for delays.
        unit_col: Column name for units.
        reference_point_col: Column name for reference points.
        target_frequency_col: Column name for frequencies.

    Returns:
        ``{entity_key: {'delays', 'delay_unit', 'reference_point', 'target_frequency'}}``,
        entity keys being tuples.
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
    """Extract a parameter of one entity: a scalar if constant, else a per-variable dict.

    Args:
        group: Rows of one entity (variable as last index level).
        column: Column holding the parameter.

    Returns:
        The single value, or ``{variable: value}``.

    Examples:
        >>> rows = pd.DataFrame({'unit': ['D', 'W']}, index=pd.Index(['GDP', 'CPI']))
        >>> _extract_param_by_variable(rows, 'unit')
        {'GDP': 'D', 'CPI': 'W'}
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
) -> Union[str, Dict[str, str]]:
    """Resolve the strategy of one entity.

    Args:
        strategy: Strategy specification (str, per-entity dict, per-variable dict or
            callable).
        entity_key: Entity identifier (tuple).

    Returns:
        ``'shift'`` or ``'mask'``, or the per-variable dict itself (dict keyed by
        variable names, applied to every entity).

    Raises:
        ValueError: If the strategy (or the callable's result) is invalid.
        KeyError: If the entity is absent from a per-entity dict.
        TypeError: If ``strategy`` is neither a string, a dict nor a callable.

    Examples:
        >>> _resolve_strategy({'FR': 'mask'}, ('FR',))
        'mask'
        >>> _resolve_strategy({'GDP': 'shift', 'CPI': 'mask'}, ('FR',))
        {'GDP': 'shift', 'CPI': 'mask'}
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
    """Prepare the ``entity_kwargs`` of a ``PanelwiseTransformer`` from a delays table.

    Alternative to :func:`create_delay_transformer_factory`: the kwargs of each
    entity are applied by ``set_params`` to a clone of a base
    ``PublicationDelayTransformer``. Entities absent from the table keep the base
    transformer (or the ``default_entity_kwargs`` of the ``PanelwiseTransformer``).

    Args:
        df_delays: Per-entity delays table (see :func:`create_delay_transformer_factory`).
        strategy: 'shift', 'mask', or a dict keyed by entity or by variable (callables
            are not supported: use the factory).
        delay_col: Column name for delays.
        unit_col: Column name for units.
        reference_point_col: Column name for reference points.
        target_frequency_col: Column name for frequencies (used by the mask only).

    Returns:
        ``{entity_key: kwargs}``, with the same configuration as the factory.

    Raises:
        ValueError: If required columns are missing or a strategy is invalid.
        KeyError: If an entity is absent from a per-entity strategy dict.

    Examples:
        >>> import pandas as pd
        >>> index = pd.MultiIndex.from_tuples([('FR', 'GDP'), ('DE', 'GDP')], names=['country', 'column'])
        >>> delays = pd.DataFrame({'delay': [45.0, 75.0], 'unit': 'D', 'frequency': 'Q',
        ...                        'reference_point': 'start'}, index=index)
        >>> kwargs = prepare_entity_kwargs_from_delays(delays, strategy={'FR': 'shift', 'DE': 'mask'})
        >>> kwargs[('DE',)]['strategy'], kwargs[('DE',)]['target_frequency'], kwargs[('FR',)]['target_frequency']
        ('mask', 'Q', None)
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

    # Conversion au format des entity_kwargs (stratégie résolue entité par entité)
    return {
        entity_key: _entity_kwargs(params, _resolve_strategy(strategy=strategy, entity_key=entity_key))
        for entity_key, params in entity_params.items()
    }


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
