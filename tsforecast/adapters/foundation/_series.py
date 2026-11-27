"""Array helpers of ``FoundationModelAdapter``, on numpy and pandas only.

``entity_grid_positions`` and ``place_on_grids`` lay the rows of each entity on its
regular grid (one array per entity, ``NaN`` on uncovered dates); ``future_windows``
lays the rows to predict on the steps after the end of training;
``entity_rank`` orders entities by label; ``check_quantile_levels``,
``bracketing_levels`` and ``interpolate_quantiles`` validate the requested
quantile levels and interpolate them from the levels a model was trained on.
"""
# Importation des modules
# Modules de base
from typing import Optional, Sequence, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd

# tsforecast
from tsforecast.adapters._base import TemporalLayout, steps_between
from tsforecast.adapters.foundation.protocol import BackendCapabilities, same_level


# Fonction de regroupement des lignes par code d'entité
def _rows_by_code(codes: np.ndarray, n_codes: int) -> Tuple[np.ndarray, ...]:
    """Split row positions by entity code with one stable sort.

    Args:
        codes: Entity code of each row, in ``[0, n_codes)``.
        n_codes: Number of entity codes.

    Returns:
        For each code, the positions of its rows in increasing order (an empty
        array for a code without rows).
    """
    # Tri stable des codes puis découpage aux frontières de chaque code
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes, minlength=n_codes))[:-1]
    return tuple(np.split(order, bounds))


# Fonction de position des lignes sur la grille de leur entité
def entity_grid_positions(layout: TemporalLayout, freq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Locate each row on the regular grid of its entity, which starts at its first date.

    Args:
        layout: Temporal and entity structure of the rows.
        freq: Pandas alias of the grid frequency.

    Returns:
        ``(positions, lengths)``: the grid step of each row (``np.int64``,
        ``0`` at the first date of its entity) and the number of grid steps
        from the first to the last date of each entity, in the order of the
        codes.

    Raises:
        ValueError: If a date is not on the grid.

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.adapters._base import TemporalLayout
        >>> index = pd.MultiIndex.from_arrays(
        ...     [['A', 'A', 'B'], pd.to_datetime(['2024-01-01', '2024-04-01', '2024-02-01'])])
        >>> positions, lengths = entity_grid_positions(TemporalLayout.from_index(index, freq='MS'), 'MS')
        >>> positions.tolist(), lengths.tolist()
        ([0, 3, 0], [4, 1])
    """
    # Première date de chaque entité, diffusée sur ses lignes
    first = pd.DatetimeIndex(pd.Series(layout.dates).groupby(layout.codes, sort=True).min())
    positions = steps_between(first[layout.codes], layout.dates, freq)
    # Longueur de grille de chaque entité : dernière position + 1
    lengths = np.zeros(layout.n_entities, dtype=np.int64)
    np.maximum.at(lengths, layout.codes, positions + 1)
    return positions, lengths


# Fonction de placement des lignes sur la grille de leur entité
def place_on_grids(
    values: np.ndarray,
    codes: np.ndarray,
    positions: np.ndarray,
    lengths: np.ndarray,
    keep: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Lay the values of each entity on its grid, keeping its last ``keep`` steps.

    Args:
        values: Values of shape ``(n_rows, n_columns)``.
        codes: Entity code of each row.
        positions: Grid step of each row on the grid of its entity.
        lengths: Number of grid steps of each entity.
        keep: Number of most recent steps to keep per entity, or None for all.

    Returns:
        One read-only ``(n_columns, n_steps)`` array per entity, oldest step
        first, ``NaN`` on the steps without a row.

    Examples:
        >>> import numpy as np
        >>> grids = place_on_grids(np.array([[1.0], [4.0], [7.0]]), np.array([0, 0, 1]),
        ...                        np.array([0, 3, 0]), np.array([4, 1]), keep=3)
        >>> [grid.tolist() for grid in grids]
        [[[nan, nan, 4.0]], [[7.0]]]
    """
    # Grilles de chaque entité, remplies à partir de ses lignes
    grids = []
    # Parcours des entités dans l'ordre de leurs codes
    for code, rows in enumerate(_rows_by_code(codes, len(lengths))):
        length = int(lengths[code])
        # Premier pas gardé : les keep derniers pas de la grille
        start = max(0, length - keep) if keep is not None else 0
        grid = np.full((values.shape[1], length - start), np.nan)
        # Lignes gardées, placées à leur pas
        kept = rows[positions[rows] >= start]
        grid[:, positions[kept] - start] = values[kept].T
        grid.setflags(write=False)
        grids.append(grid)
    return tuple(grids)


# Fonction de placement des lignes à prévoir sur les pas suivant l'entraînement
def future_windows(
    values: np.ndarray, codes: np.ndarray, steps: np.ndarray, n_entities: int
) -> Tuple[Optional[np.ndarray], ...]:
    """Lay the rows to predict of each entity on the steps after its last training date.

    Args:
        values: Values of shape ``(n_rows, n_columns)``.
        codes: Training entity code of each row.
        steps: Number of grid steps from the last training date of the
            entity of each row to its date (at least 1).
        n_entities: Number of training entities.

    Returns:
        For each training entity, a read-only ``(n_columns, max_step)`` array
        whose column ``k - 1`` holds the row ``k`` steps after the end of
        training (``NaN`` for a step without a row), or None for an entity
        without a row to predict.

    Examples:
        >>> import numpy as np
        >>> windows = future_windows(np.array([[5.0], [6.0]]), np.array([1, 1]), np.array([3, 1]), 2)
        >>> windows[0] is None, windows[1].tolist()
        (True, [[6.0, nan, 5.0]])
    """
    # Fenêtre de chaque entité, None sans ligne à prévoir
    windows = []
    # Parcours des entités dans l'ordre de leurs codes
    for rows in _rows_by_code(codes, n_entities):
        # Entité sans ligne à prévoir
        if len(rows) == 0:
            windows.append(None)
            continue
        window = np.full((values.shape[1], int(steps[rows].max())), np.nan)
        window[:, steps[rows] - 1] = values[rows].T
        window.setflags(write=False)
        windows.append(window)
    return tuple(windows)


# Fonction de rang des entités dans l'ordre de leurs étiquettes
def entity_rank(entities: Sequence[tuple]) -> np.ndarray:
    """Rank each entity in the order of its labels, so that batches do not depend on row order.

    Args:
        entities: Entity keys (tuples), in the order of their codes.

    Returns:
        ``np.intp`` rank of each entity; the order of the codes when the
        labels cannot be compared.

    Examples:
        >>> entity_rank([('B',), ('A',), ('C',)]).tolist()
        [1, 0, 2]
    """
    # Rang par défaut : ordre des codes
    rank = np.arange(len(entities), dtype=np.intp)
    # Série ou entité unique : rien à ordonner
    if len(entities) < 2:
        return rank
    try:
        order = pd.MultiIndex.from_tuples(list(entities)).argsort()
    except TypeError:
        # Étiquettes non comparables : ordre des codes gardé
        return rank
    rank[order] = np.arange(len(entities), dtype=np.intp)
    return rank


# Fonction de validation des niveaux de quantiles demandés
def check_quantile_levels(levels: Sequence[float], capabilities: BackendCapabilities, name: str) -> np.ndarray:
    """Validate requested quantile levels against the levels a model can produce.

    Args:
        levels: Requested levels, in any order.
        capabilities: Capabilities of the loaded model.
        name: Name of the argument, for the error messages.

    Returns:
        The levels as a ``float64`` array, in the given order.

    Raises:
        ValueError: If no level is given, a level is not a number of
            ``]0, 1[`` or is repeated, or a level lies outside the range of the
            levels the model was trained on.

    Examples:
        >>> from tsforecast.adapters.foundation.protocol import BackendCapabilities
        >>> check_quantile_levels([0.9, 0.1], BackendCapabilities(), 'quantile_levels').tolist()
        [0.9, 0.1]
    """
    # Conversion en tableau de flottants à une dimension
    try:
        values = np.array(levels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of numbers, got {levels!r}") from None
    # Niveau absent, hors de ]0, 1[ ou répété
    if len(values) == 0:
        raise ValueError(f"{name} must hold at least one level")
    if not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {values.tolist()}")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must not repeat a level, got {values.tolist()}")
    # Niveaux hors de la plage apprise : aucune extrapolation inventée
    native = capabilities.quantile_levels
    if native is not None:
        low, high = native[0], native[-1]
        outside = [level for level in values.tolist() if level < low - 1e-12 or level > high + 1e-12]
        if outside:
            raise ValueError(
                f"{name} {outside} lie outside the range [{low}, {high}] of the quantile levels the "
                "model was trained on"
            )
    return values


# Fonction de choix des niveaux appris encadrant les niveaux demandés
def bracketing_levels(levels: np.ndarray, native: Sequence[float]) -> Tuple[float, ...]:
    """Select the native levels needed to interpolate each requested level.

    Args:
        levels: Requested levels, within the range of ``native``.
        native: Levels the model was trained on, strictly increasing.

    Returns:
        The native levels equal to a requested level or adjacent to it,
        strictly increasing.

    Examples:
        >>> bracketing_levels(np.array([0.5, 0.15]), (0.1, 0.2, 0.5, 0.9))
        (0.1, 0.2, 0.5)
    """
    # Indices des niveaux appris utiles
    native = np.asarray(native, dtype=np.float64)
    needed = set()
    # Parcours des niveaux demandés : niveau appris égal, ou les deux qui l'encadrent
    for level in levels.tolist():
        equal = [index for index, value in enumerate(native.tolist()) if same_level(value, level)]
        if equal:
            needed.add(equal[0])
            continue
        upper = int(np.searchsorted(native, level))
        needed.update((upper - 1, upper))
    return tuple(float(native[index]) for index in sorted(needed))


# Fonction d'interpolation linéaire des quantiles
def interpolate_quantiles(values: np.ndarray, source: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Interpolate quantile forecasts linearly between levels.

    Args:
        values: Forecasts whose last axis follows ``source``.
        source: Levels of ``values``, strictly increasing.
        target: Levels to return, each within the range of ``source``.

    Returns:
        Forecasts whose last axis follows ``target``; a target level equal to
        a source level returns that source value exactly.

    Examples:
        >>> import numpy as np
        >>> interpolate_quantiles(np.array([1.0, 3.0, 7.0]), (0.25, 0.5, 0.75), (0.375, 0.75)).tolist()
        [2.0, 7.0]
    """
    # Niveaux sources et cibles en tableaux
    source = np.asarray(source, dtype=np.float64)
    result = np.empty(values.shape[:-1] + (len(target),))
    # Parcours des niveaux cibles : valeur exacte, ou combinaison des deux niveaux voisins
    for position, level in enumerate(target):
        equal = np.flatnonzero(np.abs(source - level) <= 1e-12)
        if len(equal):
            result[..., position] = values[..., equal[0]]
            continue
        upper = int(np.searchsorted(source, level))
        weight = (level - source[upper - 1]) / (source[upper] - source[upper - 1])
        result[..., position] = (1.0 - weight) * values[..., upper - 1] + weight * values[..., upper]
    return result
