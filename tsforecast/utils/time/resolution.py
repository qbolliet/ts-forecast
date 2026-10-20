"""Exact change of the storage unit of dates: ``as_exact_unit``.

pandas stores dates in seconds, milliseconds, microseconds or nanoseconds.
``as_exact_unit`` expresses dates at a requested unit without ever rounding:
dates with a finer part move to the coarsest finer unit that holds them. The
public outputs of the package use it to return their dates at the unit of
their input.
"""
# Importation des modules
from typing import TypeVar

import pandas as pd

# Unités de stockage des dates, de la plus grossière à la plus fine
from .._constants import DATETIME_UNITS

# Type des dates acceptées, rendu à l'identique
DateLike = TypeVar('DateLike', pd.Timestamp, pd.DatetimeIndex)


# Fonction d'expression exacte de dates dans une unité de stockage
def as_exact_unit(dates: DateLike, unit: str) -> DateLike:
    """Express dates at ``unit``, or at the coarsest finer unit that holds them exactly.

    The instants never change, only their storage unit does. A date with a
    part finer than ``unit`` (half a second at unit ``'s'``) is never
    rounded: the dates are then stored at the coarsest finer unit that
    represents all of them exactly (``'s'``, then ``'ms'``, ``'us'``,
    ``'ns'``). An index has a single unit, so one such date moves the whole
    index. The time zone, the name and the frequency of an index are kept.

    Args:
        dates: A ``pd.Timestamp`` or a ``pd.DatetimeIndex`` (``NaT`` values
            and empty indexes accepted).
        unit: Target unit, one of ``'s'``, ``'ms'``, ``'us'`` and ``'ns'``.

    Returns:
        The same instants, with the same type, at ``unit`` or at the
        coarsest finer unit holding them exactly.

    Raises:
        ValueError: If ``unit`` is not one of the four pandas units.
        TypeError: If ``dates`` is neither a ``pd.Timestamp`` nor a
            ``pd.DatetimeIndex``.
        pandas.errors.OutOfBoundsDatetime: If a date lies outside the range
            of every unit able to hold it exactly (in practice, outside the
            nanosecond range, 1677-09-21 to 2262-04-11, for ``unit='ns'``).

    Examples:
        >>> import pandas as pd
        >>> grid = pd.date_range('2024-01-01', periods=3, freq='MS')
        >>> grid.unit
        'us'
        >>> as_exact_unit(grid, 's').unit
        's'
        >>> half_second = pd.Timestamp('2024-01-01 00:00:00.5')
        >>> as_exact_unit(half_second, 's').unit
        'ms'
        >>> bool(as_exact_unit(half_second, 's') == half_second)
        True
    """
    # Vérification des arguments
    if unit not in DATETIME_UNITS:
        raise ValueError(f"unit must be one of {DATETIME_UNITS}, got {unit!r}")
    if not isinstance(dates, (pd.Timestamp, pd.DatetimeIndex)):
        raise TypeError(
            f"dates must be a pd.Timestamp or a pd.DatetimeIndex, got {type(dates).__name__}"
        )

    # Unités candidates, de l'unité demandée vers les plus fines : la première
    # qui représente exactement toutes les dates est retenue, sans arrondi
    candidates = DATETIME_UNITS[DATETIME_UNITS.index(unit):]
    for candidate in candidates[:-1]:
        try:
            return dates.as_unit(candidate, round_ok=False)
        except ValueError:
            # Conversion avec perte : unité plus fine suivante. Un dépassement de
            # plage (OutOfBoundsDatetime, sous-classe de ValueError) se retrouve
            # dans les unités plus fines, de plage plus étroite, et ressort de la
            # conversion finale en nanosecondes
            continue

    # Nanoseconde : conversion toujours exacte, seule la plage peut manquer
    return dates.as_unit(candidates[-1], round_ok=False)
