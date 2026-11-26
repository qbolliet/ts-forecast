"""Cross-validation folds of temporal hierarchies: ``period_splits``.

A temporal hierarchy in long layout holds one row per node and per period of
each temporal level (twelve monthly rows, four quarterly rows and one yearly row
a year): the splitters of ``tsforecast.crossvals``, which count test rows per
entity, cannot cut it into complete periods. ``period_splits`` cuts it on the
calendar periods of the most aggregated temporal level, the ones
``HierarchicalForecastAdapter`` reconciles, and returns folds for the ``cv``
argument of scikit-learn.
"""
# Importation des modules
# Modules de base
from numbers import Integral
from typing import Any, List, Mapping, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd

# tsforecast
from tsforecast.adapters._base import TemporalLayout, steps_between
from tsforecast.adapters.hierarchicalforecast._temporal import (
    TemporalCalendar,
    detect_fine_frequency,
    parse_temporal_hierarchy,
    read_temporal_index,
)


# Fonction auxiliaire de contrôle d'un entier
def _check_integer(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` as an ``int`` if it is an integer at least ``minimum``.

    Args:
        value: Value of the parameter.
        name: Name of the parameter, for the error message.
        minimum: Smallest accepted value.

    Returns:
        ``value`` converted to ``int``.

    Raises:
        ValueError: If ``value`` is a boolean, not an integer, or below ``minimum``.
    """
    # Refus des booléens, des non-entiers et des valeurs trop petites
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


# Fonction de découpage en plis par périodes du niveau temporel le plus agrégé
def period_splits(
    X: Any,
    temporal_hierarchy: Mapping[str, int],
    *,
    n_splits: int = 2,
    test_periods: int = 1,
    gap: int = 0,
    temporal_level: Any = None,
    temporal_origin: Any = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Cut a temporal hierarchy into folds of complete periods of its most aggregated level.

    The periods are those of ``HierarchicalForecastAdapter`` with the same
    ``temporal_hierarchy``, ``temporal_level`` and ``temporal_origin`` (calendar
    years for a monthly hierarchy ``{'year': 12, ...}``). The ``n_splits``
    folds test adjacent blocks of ``test_periods`` periods ending with the last
    period of ``X``: every row of these periods (every temporal level, node
    and series) is tested. A fold trains on the rows whose node ends at least
    ``gap + 1`` fine steps before the first fine step of the test, so that an
    aggregate overlapping the test (or the gap) never trains.

    Args:
        X: Data indexed like the input of ``HierarchicalForecastAdapter``:
            ``(..., temporal level, date)``.
        temporal_hierarchy: Temporal hierarchy ``{name: number of fine
            steps}``.
        n_splits: Number of folds.
        test_periods: Number of periods of the most aggregated level tested
            per fold.
        gap: Number of fine steps left out between training and test (``h + d
            - 1`` for rows indexed by forecast origin).
        temporal_level: Name or position of the index level holding the
            temporal level of each row; None for the level before the date.
        temporal_origin: Start of one period of the most aggregated level;
            None for the Unix epoch.

    Returns:
        List of ``(train, test)`` integer position arrays, in the row order of
        ``X``, usable as ``cv`` by ``cross_val_score`` or ``GridSearchCV``.

    Raises:
        TypeError: If the dates of ``X`` are not ``datetime64``.
        ValueError: If a parameter is invalid, if ``X`` does not carry the
            temporal hierarchy, if it holds fewer periods than the folds need,
            or if a fold has no training row.

    Examples:
        >>> import pandas as pd
        >>> quarters = pd.date_range('2022-01-01', periods=12, freq='QS')
        >>> years = quarters[::4]
        >>> index = pd.MultiIndex.from_arrays(
        ...     [['quarter'] * 12 + ['year'] * 3, quarters.append(years)], names=['level', 'date'])
        >>> X = pd.DataFrame({'model': 1.0}, index=index)
        >>> folds = period_splits(X, {'year': 4, 'quarter': 1}, n_splits=2)
        >>> [(train.tolist(), test.tolist()) for train, test in folds]
        [([0, 1, 2, 3, 12], [4, 5, 6, 7, 13]), ([0, 1, 2, 3, 4, 5, 6, 7, 12, 13], [8, 9, 10, 11, 14])]
    """
    # Vérification des paramètres du découpage
    n_splits = _check_integer(n_splits, "n_splits", 1)
    test_periods = _check_integer(test_periods, "test_periods", 1)
    gap = _check_integer(gap, "gap", 0)
    # Analyse de l'index : entités, dates, niveau temporel de chaque ligne
    index = X.index if isinstance(X, (pd.Series, pd.DataFrame)) else X
    layout = TemporalLayout.from_index(index, detect_freq=False)
    spec = parse_temporal_hierarchy(temporal_hierarchy)
    temporal_index = read_temporal_index(layout, spec, temporal_level)
    # Calendrier des périodes sur la grille fine
    freq = detect_fine_frequency(layout, temporal_index.row_levels, spec)
    calendar = TemporalCalendar.build(spec, freq, temporal_origin)
    # Période et dernier pas fin de chaque ligne
    placement = calendar.locate(temporal_index.row_levels, layout.dates)
    # Code de la période de chaque ligne, périodes triées chronologiquement
    period_codes, periods = pd.factorize(placement.period_start, sort=True)
    # Vérification du nombre de périodes : une au moins avant les périodes de test
    needed = n_splits * test_periods
    if len(periods) <= needed:
        raise ValueError(
            f"{n_splits} fold(s) of {test_periods} period(s) need more than {needed} periods of the most "
            f"aggregated temporal level {spec.levels[0]!r}, X holds {len(periods)}"
        )

    # Plis, dans l'ordre chronologique de leurs périodes de test
    folds = []
    # Parcours des plis : blocs adjacents de périodes finissant à la dernière période
    for fold in range(n_splits):
        # Code de la première période de test du pli
        first = len(periods) - (n_splits - fold) * test_periods
        # Lignes de test : toutes les lignes des périodes du bloc
        test = np.flatnonzero((period_codes >= first) & (period_codes < first + test_periods))
        # Premier pas fin du test : première date de la grille fine dans la première période
        test_start = pd.date_range(periods[first], periods=1, freq=freq)[0]
        # Pas fins entre la fin du nœud de chaque ligne et le début du test
        steps = steps_between(placement.last, pd.DatetimeIndex([test_start] * len(placement.last)), freq)
        # Lignes d'entraînement : nœud terminé au moins gap + 1 pas avant le test
        train = np.flatnonzero(steps > gap)
        if len(train) == 0:
            raise ValueError(
                f"Fold {fold} has no training row: X ends its first periods less than gap + 1 = {gap + 1} fine "
                "steps before the test"
            )
        folds.append((train, test))
    return folds
