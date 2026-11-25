"""Shared builders of the ``AggregationConstraint`` tests (not a test module).

Monthly and panel grids, raw predictions, annual observations, anchor observations and
warning recorders used by several test files of the package.
"""

# Modules de base
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd


# Seuil d'exactitude de la contrainte : un écart supérieur est une anomalie,
# jamais une tolérance à élargir
_EXACT = 1e-9

# Prédictions brutes des douze mois de 2021 : somme 112.5, décembre 10.5.
# Valeurs de l'exemple de référence — ratio 120 / 112.5, donc
# janvier 9.6, février 10.1333… et l'ancre 11.2
_RAW_2021 = [9.0, 9.5, 9.5, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 10.5]

# Alias pandas d'une grille selon sa base et sa position dans la période
_GRID_ALIASES = {
    ('M', 'start'): 'MS', ('M', 'end'): 'ME',
    ('Q', 'start'): 'QS', ('Q', 'end'): 'QE',
    ('Y', 'start'): 'YS', ('Y', 'end'): 'YE',
}


# Fabrique de la grille mensuelle d'une année
def _monthly_grid(year: int = 2021) -> pd.DatetimeIndex:
    """Build the twelve month-end dates of one year.

    Args:
        year: Calendar year of the grid.

    Returns:
        Month-end ``DatetimeIndex`` of twelve rows.
    """
    return pd.date_range(f'{year}-01-31', periods=12, freq='ME')


# Fabrique des prédictions brutes d'une année
def _raw_predictions(year: int = 2021, values=None) -> pd.Series:
    """Build the raw sub-period predictions of one year.

    Args:
        year: Calendar year of the grid.
        values: Values to carry; defaults to the reference vector.

    Returns:
        ``Series`` named ``'a1'`` on the twelve month-end dates of ``year``.
    """
    return pd.Series(
        _RAW_2021 if values is None else values,
        index=_monthly_grid(year),
        name='a1',
    )


# Fabrique des observations annuelles sur une grille mensuelle
def _annual_observations(year: int = 2021, total: float = 120.0) -> pd.Series:
    """Build the annual observation of one year, on its monthly grid.

    Args:
        year: Calendar year of the grid.
        total: Observed annual total, carried by the December anchor.

    Returns:
        ``Series`` named ``'a1'``, non-null on the December anchor only.
    """
    observations = pd.Series(np.nan, index=_monthly_grid(year), name='a1')
    observations.iloc[-1] = total
    return observations


# Fabrique d'un bloc de panel à partir d'une série temporelle
def _to_panel(series: pd.Series, entities=('FR', 'DE')) -> pd.Series:
    """Replicate a time series over several entities, as a panel Series.

    Args:
        series: Time series to replicate.
        entities: Entity keys of the panel.

    Returns:
        ``Series`` on a ``MultiIndex`` ``(country, date)``.
    """
    blocks = []
    for entity in entities:
        block = series.copy()
        block.index = pd.MultiIndex.from_product(
            [[entity], series.index], names=['country', 'date']
        )
        blocks.append(block)
    return pd.concat(blocks)


# Capture de tous les avertissements d'un appel
def _record_warnings(function, *args, **kwargs):
    """Call ``function`` and return its result with every warning raised.

    Args:
        function: Callable to run.
        *args: Positional arguments of the call.
        **kwargs: Keyword arguments of the call.

    Returns:
        Tuple ``(result, user_warnings)`` — the list of ``UserWarning``
        messages, as strings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    return result, messages
