"""Flat metric builders for fitted tsforecast components.

Each function reads the fitted state of a component and returns a flat mapping of
scalar metrics, ready for ``mlflow.log_metrics``. Non-finite values (``inf`` from
a fully failed cross-validation, ``nan``) are dropped from the output, since most
trackers reject them; the associated ``*.n_failed`` / ``*.n`` counts keep that
information visible.
"""
# Importation des modules
# Modules de base
import math
from typing import Any, Dict, Iterable, List, Optional

# Manipulation de données
import numpy as np
import pandas as pd


# Fonction utilitaire de filtrage des valeurs non finies
def _finite(value: Any) -> Optional[float]:
    """Return ``float(value)`` when finite, ``None`` otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# Fonction utilitaire de résumé d'une collection de nombres
def _summarize(values: Iterable[Any], prefix: str) -> Dict[str, float]:
    """Build ``{prefix}.{min,max,mean,n}`` over the finite values of ``values``.

    ``n`` is always emitted (0 when nothing is finite); the other three keys
    appear only when at least one value is finite.
    """
    finite = [v for v in (_finite(x) for x in values) if v is not None]
    out: Dict[str, float] = {f"{prefix}.n": float(len(finite))}
    if finite:
        out[f"{prefix}.min"] = float(np.min(finite))
        out[f"{prefix}.max"] = float(np.max(finite))
        out[f"{prefix}.mean"] = float(np.mean(finite))
    return out


# Fonction utilitaire d'aplatissement d'un bloc de statistiques de provenance
def _flatten_provenance_block(block: Dict[str, Any], prefix: str) -> Dict[str, float]:
    """Flatten one ``compute_statistics`` block into ``{prefix}.<key>`` floats."""
    flat: Dict[str, float] = {}
    for key, value in block.items():
        number = _finite(value)
        if number is not None:
            flat[f"{prefix}.{key}"] = number
    return flat


# Métriques d'un imputeur multi-fréquences ajusté
def imputation_metrics(
    imputer: Any,
    *,
    per_column: bool = False,
) -> Dict[str, float]:
    """Summarize a fitted mixed-frequency imputer as flat tracking metrics.

    Works with a fitted :class:`~tsforecast.frequency.HighFrequencyImputer`.
    The provenance
    statistics are read from ``provenance_statistics_`` when present (the last
    ``transform``), otherwise from ``provenance_statistics_fit_`` (the ``fit``).

    Args:
        imputer: A fitted imputer exposing ``provenance_statistics_`` or
            ``provenance_statistics_fit_``.
        per_column: When ``True``, also emit ``<column>.provenance.<key>`` for
            every column of the provenance breakdown, not only the overall
            block.

    Returns:
        Mapping ``{metric_name: float}``. Keys: ``provenance.<key>`` (counts and
        percentages per ``ProvenanceType`` over the whole matrix, e.g.
        ``provenance.original_pct``, ``provenance.model_on_true``);
        ``n_stages`` (frequencies in the cascade); ``n_unanchored_pairs``
        (``(entity, column)`` pairs imputed with no anchor); ``cv_score.{min,max,mean,n}`` (cross-validated ordering scores,
        only when the ``'cv'`` ordering ran with a non-empty result); and
        ``<column>.provenance.<key>`` per column when ``per_column=True``.

    Raises:
        AttributeError: If ``imputer`` exposes neither ``provenance_statistics_``
            nor ``provenance_statistics_fit_`` (i.e. it is not fitted, or not an
            imputer).

    Examples:
        >>> metrics = imputation_metrics(fitted_imputer)   # doctest: +SKIP
        >>> import mlflow                                   # doctest: +SKIP
        >>> mlflow.log_metrics(metrics)                     # doctest: +SKIP
    """
    # Résolution du bloc de statistiques : transform prioritaire, puis fit
    statistics = getattr(imputer, "provenance_statistics_", None)
    if statistics is None:
        statistics = getattr(imputer, "provenance_statistics_fit_", None)
    if statistics is None:
        raise AttributeError(
            "imputer exposes neither 'provenance_statistics_' nor "
            "'provenance_statistics_fit_'; pass a fitted HighFrequencyImputer."
        )

    metrics: Dict[str, float] = {}

    # Bloc global de provenance
    overall = statistics.get("overall", {})
    metrics.update(_flatten_provenance_block(overall, "provenance"))

    # Blocs par colonne, optionnels
    if per_column:
        for name, block in statistics.items():
            if name == "overall":
                continue
            metrics.update(_flatten_provenance_block(block, f"{name}.provenance"))

    # Résumé de la cascade
    progression = getattr(imputer, "frequency_progression_", None)
    if progression is not None:
        metrics["n_stages"] = float(len(progression))
    metrics["n_unanchored_pairs"] = float(
        len(getattr(imputer, "unanchored_pairs_", ()) or ())
    )

    # Scores de validation croisée d'ordonnancement
    cv_scores = getattr(imputer, "imputation_cv_scores_", None)
    if cv_scores:
        flat_scores: List[Any] = [
            score
            for stage_scores in cv_scores.values()
            for score in stage_scores.values()
        ]
        metrics.update(_summarize(flat_scores, "cv_score"))

    return metrics


# Métriques d'un transformateur de délais de publication ajusté
def delay_metrics(delay_transformer: Any) -> Dict[str, float]:
    """Summarize the delays applied by a fitted ``PublicationDelayTransformer``.

    Args:
        delay_transformer: A fitted
            :class:`~tsforecast.delays.PublicationDelayTransformer` exposing
            ``shift_params`` / ``mask_params``.

    Returns:
        Mapping ``{metric_name: float}``. Keys: ``n_shift_columns`` /
        ``n_mask_columns`` / ``n_delayed_columns`` (columns touched by each
        strategy); ``shift_periods.{min,max,mean,n}`` (periods shifted, over the
        shift columns); ``mask_obs.{min,max,mean,n}`` (observations masked, over
        the mask columns); ``delay.{min,max,mean,n}`` (raw delay values from the
        ``delays`` specification, in their original unit).

    Raises:
        AttributeError: If ``delay_transformer`` is not fitted (no
            ``shift_params`` / ``mask_params``).

    Examples:
        >>> delay_metrics(fitted_delay_transformer)         # doctest: +SKIP
        {'n_shift_columns': 2.0, ...}
    """
    # Vérification de l'ajustement
    if not hasattr(delay_transformer, "shift_params") and not hasattr(
        delay_transformer, "mask_params"
    ):
        raise AttributeError(
            "delay_transformer is not fitted: no 'shift_params' / 'mask_params'."
        )

    shift_params: Dict[str, Any] = getattr(delay_transformer, "shift_params", {}) or {}
    mask_params: Dict[str, Any] = getattr(delay_transformer, "mask_params", {}) or {}

    metrics: Dict[str, float] = {
        "n_shift_columns": float(len(shift_params)),
        "n_mask_columns": float(len(mask_params)),
        "n_delayed_columns": float(len(shift_params) + len(mask_params)),
    }

    # Périodes décalées et observations masquées
    metrics.update(
        _summarize(
            (params.get("n_periods") for params in shift_params.values()),
            "shift_periods",
        )
    )
    metrics.update(
        _summarize(
            (params.get("n_obs") for params in mask_params.values()),
            "mask_obs",
        )
    )

    # Valeurs de délai brutes issues de la spécification
    delays = getattr(delay_transformer, "delays", None)
    raw_values: Iterable[Any]
    if isinstance(delays, pd.DataFrame) and "delay" in delays.columns:
        raw_values = delays["delay"].tolist()
    elif isinstance(delays, dict):
        raw_values = list(delays.values())
    else:
        raw_values = []
    metrics.update(_summarize(raw_values, "delay"))

    return metrics


# Résumé des plis d'un splitter de validation croisée
def split_summary(
    splitter: Any,
    X: Any,
    y: Any = None,
    groups: Any = None,
) -> Dict[str, float]:
    """Summarize the folds a tsforecast splitter produces on ``X``.

    Args:
        splitter: Any splitter from :mod:`tsforecast.crossvals` (or any object
            exposing an sklearn-style ``split(X, y, groups)``).
        X: Data passed to ``split`` — array-like or DataFrame.
        y: Optional target passed to ``split``.
        groups: Optional groups passed to ``split`` (entities, for panel
            splitters that require them).

    Returns:
        Mapping ``{metric_name: float}``: ``n_splits``, ``train_size.{min,max,
        mean}``, ``test_size.{min,max,mean}``, and ``gap`` when the splitter
        carries one.

    Examples:
        >>> import numpy as np
        >>> from tsforecast.crossvals import TSOutOfSampleSplit
        >>> X = np.arange(60).reshape(-1, 1)
        >>> summary = split_summary(
        ...     TSOutOfSampleSplit(n_splits=3, test_size=5, gap=1), X
        ... )
        >>> summary["n_splits"]
        3.0
        >>> summary["test_size.max"]
        5.0
        >>> summary["gap"]
        1.0
    """
    # Collecte des tailles de chaque pli
    train_sizes: List[int] = []
    test_sizes: List[int] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))

    metrics: Dict[str, float] = {"n_splits": float(len(train_sizes))}
    # Les clés min/max/mean n'ont de sens qu'avec au moins un pli
    if train_sizes:
        metrics["train_size.min"] = float(np.min(train_sizes))
        metrics["train_size.max"] = float(np.max(train_sizes))
        metrics["train_size.mean"] = float(np.mean(train_sizes))
        metrics["test_size.min"] = float(np.min(test_sizes))
        metrics["test_size.max"] = float(np.max(test_sizes))
        metrics["test_size.mean"] = float(np.mean(test_sizes))

    # Écart train/test, lorsque le splitter en porte un
    gap = _finite(getattr(splitter, "gap", None))
    if gap is not None:
        metrics["gap"] = gap

    return metrics
