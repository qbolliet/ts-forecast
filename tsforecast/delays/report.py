"""Structured reports of the publication delays components.

Two immutable reports gather everything a production user may want to log (in
plain logs or in an experiment tracker) when using the ``delays`` module:

- :class:`DelayDetectionReport`: what :func:`compare_and_detect_delays` saw and
  decided (returned with ``return_report=True``);
- :class:`DelayFitReport`: what :class:`PublicationDelayTransformer` resolved at
  ``fit`` time (fitted attribute ``fit_report_``).

The reports are plain data, with no dependency on any tracker. They render as a
one-line :meth:`summary` (for logs), a JSON-compatible :meth:`to_dict` (for
``mlflow.log_dict``) and a :meth:`to_frame` (for CSV artifacts). Flat numeric
metrics are built by :mod:`tsforecast.tracking`.
"""
# Importation des modules
# Modules de base
import math
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Manipulation de données
import numpy as np
import pandas as pd


# Fonction utilitaire de conversion d'une clé de variable en chaîne (clés JSON)
def _key_to_str(key: Any) -> str:
    """Render a variable key as a string.

    Args:
        key: Variable key, either a scalar (column name) or a tuple such as
            ``(entity, column)``.

    Returns:
        The key as a string; tuple parts are joined by ``'/'``.

    Examples:
        >>> _key_to_str(('FR', 'GDP'))
        'FR/GDP'
        >>> _key_to_str('GDP')
        'GDP'
    """
    if isinstance(key, tuple):
        return "/".join(str(part) for part in key)
    return str(key)


# Fonction utilitaire de conversion récursive en structure compatible JSON
def _to_jsonable(value: Any) -> Any:
    """Convert a report value into a JSON-compatible structure.

    Dict keys become strings (tuples joined by ``'/'``), tuples and sets become lists,
    timestamps ISO strings, numpy scalars Python numbers and non-finite floats ``None``.

    Args:
        value: Report value to convert (possibly nested).

    Returns:
        JSON-compatible equivalent of ``value``; unsupported types are rendered
        with ``str``.

    Examples:
        >>> _to_jsonable({('FR', 'GDP'): (1, float('nan'))})
        {'FR/GDP': [1, None]}
    """
    if isinstance(value, dict):
        return {_key_to_str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


# Fonction utilitaire de statistiques descriptives d'une série de délais
def delay_statistics(delays: pd.Series) -> Dict[str, Any]:
    """Summarize a series of delays, ignoring the unknown (``NaN``) ones.

    Args:
        delays: Delay values, in a common unit.

    Returns:
        Dict with ``n_known`` and ``n_negative`` (integers) and ``min``, ``max``,
        ``mean``, ``median`` (floats, ``None`` when no delay is known).

    Examples:
        >>> stats = delay_statistics(pd.Series([10.0, -2.0, np.nan, 30.0]))
        >>> stats['n_known'], stats['n_negative'], stats['median']
        (3, 1, 10.0)
        >>> delay_statistics(pd.Series([np.nan]))['min'] is None
        True
    """
    # Conversion des délais en valeurs numériques
    known = pd.to_numeric(delays, errors='coerce').dropna().to_numpy(dtype=float)
    # Initialisation du dictionnaire de statistique
    stats: Dict[str, Any] = {
        'n_known': int(known.size),
        'n_negative': int((known < 0).sum()),
        'min': None, 'max': None, 'mean': None, 'median': None,
    }
    # Population du dictionnaire de statistique
    if known.size:
        stats.update(
            min=float(known.min()), max=float(known.max()),
            mean=float(known.mean()), median=float(np.median(known)),
        )
    return stats


# Classe de base des rapports
class _Report:
    """Behaviour shared by the reports: JSON-compatible dict, summary, frame.

    Subclasses must be dataclasses and implement :meth:`summary` and
    :meth:`to_frame`.
    """

    # Rendu en dictionnaire compatible JSON
    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-compatible dict.

        Dict keys are strings (a ``(entity, column)`` key becomes ``'entity/column'``),
        tuples are lists, dates ISO strings and non-finite floats ``None``.

        Returns:
            Nested dict that ``json.dumps`` (or ``mlflow.log_dict``) accepts.

        Examples:
            >>> import json
            >>> json.dumps(report.to_dict())   # doctest: +SKIP
        """
        return _to_jsonable(asdict(self))  # type: ignore[call-overload]

    # Rendu en une ligne
    def summary(self) -> str:  # pragma: no cover - implémenté par les sous-classes
        """Return a one-line description of the report, meant for logs.

        Returns:
            One-line string.

        Raises:
            NotImplementedError: Always, subclasses must override it.
        """
        raise NotImplementedError

    # Rendu en jeu de données
    def to_frame(self) -> pd.DataFrame:  # pragma: no cover - implémenté par les sous-classes
        """Return the report as a DataFrame, meant for CSV artifacts.

        Returns:
            DataFrame view of the report.

        Raises:
            NotImplementedError: Always, subclasses must override it.
        """
        raise NotImplementedError

    # Représentation sous forme de chaîne de caractères
    def __str__(self) -> str:
        return self.summary()


# Rapport de détection des délais de publication
@dataclass(frozen=True)
class DelayDetectionReport(_Report):
    """What :func:`~tsforecast.delays.compare_and_detect_delays` saw and decided.

    Attributes:
        detection_mode: ``'new_only'`` or ``'all_changes'``.
        reference_point: ``'start'`` or ``'end'``.
        delay_unit: Label of the delay unit (``'day'``, ``'second'``, ``'microsecond'``).
        download_date: Resolved download date.
        has_existing_data: Whether ``existing_data`` was given. Without it, the latest
            observation of each (entity, column) couple is detected and counted
            as a new value.
        n_rows_new: Number of rows of ``new_data``.
        n_rows_existing: Number of rows of ``existing_data`` (``None`` without it).
        n_entities: Number of panel entities (0 for a time series).
        n_columns: Number of variables of ``new_data``.
        columns_compared: Variables present in both datasets (all of ``new_data``
            without ``existing_data``).
        columns_new_only: Variables of ``new_data`` absent from ``existing_data``,
            hence not compared (never reported).
        columns_existing_only: Variables of ``existing_data`` absent from ``new_data``.
        n_detected: Number of detected observations (rows of the result).
        n_new_values: Detected null → non-null transitions.
        n_revisions: Detected revisions of a value (``'all_changes'`` mode only).
        n_vanished_values: Values non-null in ``existing_data`` that are ``NaN`` or
            missing in ``new_data``. Neither mode reports them.
        n_detected_by_column: Number of detected observations per variable.
        columns_without_detection: Compared variables with no detected observation.
        frequencies: Detected frequency (literal) of each variable, or of each
            ``(entity, ..., column)`` key for a panel; ``None`` when undetectable.
            Empty when nothing is detected.
        undetected_keys: Keys of the detected observations whose frequency could not be
            detected (their period bounds and delay are ``NaN``).
        n_known: Number of detected observations with a known delay.
        n_negative: Number of negative delays among them.
        delay_min: Smallest known delay, in ``delay_unit`` (``None`` if none is known).
        delay_max: Largest known delay.
        delay_mean: Mean of the known delays.
        delay_median: Median of the known delays.
    """

    # Initialisation des attributs
    detection_mode: str
    reference_point: str
    delay_unit: str
    download_date: datetime
    has_existing_data: bool
    n_rows_new: int
    n_rows_existing: Optional[int]
    n_entities: int
    n_columns: int
    columns_compared: Tuple[Any, ...]
    columns_new_only: Tuple[Any, ...]
    columns_existing_only: Tuple[Any, ...]
    n_detected: int
    n_new_values: int
    n_revisions: int
    n_vanished_values: int
    n_detected_by_column: Dict[Any, int]
    columns_without_detection: Tuple[Any, ...]
    frequencies: Dict[Any, Optional[str]]
    undetected_keys: Tuple[Any, ...]
    n_known: int
    n_negative: int
    delay_min: Optional[float]
    delay_max: Optional[float]
    delay_mean: Optional[float]
    delay_median: Optional[float]

    # Méthode de résumé de la détection des délais
    def summary(self) -> str:
        """Return a one-line description of the detection.

        Returns:
            One-line string with the mode, the detection counters and the delay
            statistics (when at least one delay is known).

        Examples:
            >>> report.summary()   # doctest: +SKIP
            'delays detection: 2 observations detected (2 new, 0 revised) over 3 columns, delay in day: min=75.0 max=75.0 mean=75.0, 0 vanished, 0 undetected frequency'
        """
        # Initialisation du message
        text = (
            f"delays detection ({self.detection_mode}, reference={self.reference_point}): "
            f"{self.n_detected} observations detected ({self.n_new_values} new, "
            f"{self.n_revisions} revised) over {len(self.columns_compared)} compared columns"
        )
        # Ajout des informations sur les délais calculés si elles sont renseignées
        if self.delay_min is not None:
            text += (
                f", delay in {self.delay_unit}: min={self.delay_min:g} "
                f"max={self.delay_max:g} mean={self.delay_mean:g}"
            )
        # Fin du message
        text += (
            f", {self.n_vanished_values} vanished values, "
            f"{len(self.undetected_keys)} undetected frequencies"
        )
        return text

    # Conversion du rapport en DataFrame
    def to_frame(self) -> pd.DataFrame:
        """Return the per-variable view: detections, frequency and comparison status.

        Returns:
            DataFrame with one row per variable of ``new_data`` (index: ``'column'``) and
            the columns ``n_detected`` and ``compared``. The frequencies, which can differ
            per entity, are in :attr:`frequencies`.

        Examples:
            >>> report.to_frame()   # doctest: +SKIP
                    n_detected  compared
            column
            GDP              2      True
        """
        # Colonnes du jeu de données
        columns = list(self.columns_compared) + list(self.columns_new_only)
        return pd.DataFrame(
            {
                'n_detected': [self.n_detected_by_column.get(col, 0) for col in columns],
                'compared': [col in self.columns_compared for col in columns],
            },
            index=pd.Index(columns, name='column', dtype=object),
        )


# Enregistrement du paramétrage résolu d'une variable
@dataclass(frozen=True)
class ColumnDelayRecord:
    """Resolved delay setting of one variable of a :class:`DelayFitReport`.

    Attributes:
        column: Variable name.
        strategy: Effective strategy, ``'shift'`` or ``'mask'`` (a column moved from
            mask to shift reports ``'shift'`` and ``moved_from_mask=True``).
        delay: Raw delay value of the ``delays`` specification.
        delay_unit: Unit of the delay, as resolved.
        reference_point: ``'start'`` or ``'end'``, as resolved.
        frequency: Detected frequency of the variable (base code).
        n_periods: Number of periods shifted (``'shift'`` strategy).
        n_obs: Number of observations masked (``'mask'`` strategy).
        target_frequency: Frequency of the masked periods (``'mask'`` strategy).
        delay_unit_source: Origin of the unit: ``'explicit'`` (constructor argument),
            ``'inferred'`` (``delays`` DataFrame), ``'default'`` (``default_values``)
            or ``None`` (not resolved).
        reference_point_source: Origin of the reference point, same values.
        target_frequency_source: Origin of the target frequency, same values.
        moved_from_mask: True when the masking was impossible and the column was shifted.
    """
    # Instanciation des attributs
    column: Any
    strategy: str
    delay: Optional[float]
    delay_unit: Optional[str]
    reference_point: Optional[str]
    frequency: Optional[str]
    n_periods: Optional[int]
    n_obs: Optional[int]
    target_frequency: Optional[str]
    delay_unit_source: Optional[str]
    reference_point_source: Optional[str]
    target_frequency_source: Optional[str]
    moved_from_mask: bool


# Rapport d'ajustement du transformateur de délais
@dataclass(frozen=True)
class DelayFitReport(_Report):
    """What :class:`~tsforecast.delays.PublicationDelayTransformer` resolved at ``fit``.

    Attributes:
        prediction_date: Resolved prediction date.
        columns: One :class:`ColumnDelayRecord` per delayed variable.
        columns_unaffected: Variables of ``X`` left untouched (no delay applies).
        columns_ignored: Variables of the ``delays`` specification absent from ``X``.
        defaults_imputed: ``(column, parameter)`` pairs filled from ``default_values``.
        mask_fallbacks: Variables whose masking was impossible and that were shifted instead.
    """
    # Instanciation des attributs
    prediction_date: datetime
    columns: Tuple[ColumnDelayRecord, ...]
    columns_unaffected: Tuple[Any, ...]
    columns_ignored: Tuple[Any, ...]
    defaults_imputed: Tuple[Tuple[Any, str], ...]
    mask_fallbacks: Tuple[Any, ...]

    # Méthode de résumé
    def summary(self) -> str:
        """Return a one-line description of the fit.

        Returns:
            One-line string with the numbers of shifted, masked, unaffected and
            ignored variables, of imputed defaults and of mask fallbacks.

        Examples:
            >>> pdt.fit(X).fit_report_.summary()   # doctest: +SKIP
            'delays fit (prediction_date=2024-12-15): 2 shifted, 0 masked, 1 unaffected, 0 ignored, 0 defaults imputed, 0 mask fallbacks'
        """
        # Comptage des délais shiftés et masqués
        n_masked = sum(record.strategy == 'mask' for record in self.columns)
        n_shifted = len(self.columns) - n_masked
        return (
            f"delays fit (prediction_date={self.prediction_date:%Y-%m-%d}): "
            f"{n_shifted} shifted, {n_masked} masked, {len(self.columns_unaffected)} unaffected, "
            f"{len(self.columns_ignored)} ignored, {len(self.defaults_imputed)} defaults imputed, "
            f"{len(self.mask_fallbacks)} mask fallbacks"
        )

    # Méthode de conversion en DataFrame
    def to_frame(self) -> pd.DataFrame:
        """Return the per-variable settings, one row per delayed variable.

        Returns:
            DataFrame indexed by ``'column'`` with the fields of :class:`ColumnDelayRecord`.

        Examples:
            >>> pdt.fit(X).fit_report_.to_frame()   # doctest: +SKIP
        """
        names: List[str] = [f.name for f in fields(ColumnDelayRecord) if f.name != 'column']
        return pd.DataFrame(
            [{name: getattr(record, name) for name in names} for record in self.columns],
            columns=names,
            index=pd.Index([record.column for record in self.columns], name='column', dtype=object),
        )
