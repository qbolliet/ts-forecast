"""Shared builders of the ``CovariateMaterializer`` tests (not a test module).

Detected-frequency mappings of the reference datasets, grid builders, a
public-path way of priming the three stores, and a minimal double of the
``AggregationConstraintApplier`` protocol.
"""
# Modules de base
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# Manipulation de données
import pandas as pd

# Objet testé
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer


# Fréquences détectées du jeu TS
TS_FREQUENCIES: Dict[str, Any] = {'m1': 'M', 'q1': 'Q', 'a1': 'Y', 'a2': 'Y'}

# Fréquences du jeu PANEL : celles de TS, plus l'enquête mensuelle.
# climat_affaires est mensuelle pour FR et DE et jamais observée par IT : la
# forme scalaire est celle que rend l'adaptateur de l'imputeur quand toutes
# les entités qui observent la colonne s'accordent
PANEL_FREQUENCIES: Dict[str, Any] = {**TS_FREQUENCIES, 'climat_affaires': 'M'}

# Fréquences du jeu PANEL-F : v annuelle pour FR, trimestrielle pour DE,
# mensuelle pour IT
PANEL_F_FREQUENCIES: Dict[str, Any] = {
    'm1': 'M',
    'q1': 'Q',
    'v': {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
}

# Fréquences de heterogeneous_coverage_panel, par construction de
# HETEROGENEOUS_PANEL_COUNTRIES : dépenses publiques annuelles pour la France
# et l'Italie, trimestrielles pour l'Allemagne ; climat_affaires jamais
# observée par l'Italie (forme scalaire, comme ci-dessus). Valeurs identiques
# à celles que produisent "detect_dataset_frequency" puis l'adaptateur de
# l'imputeur sur ce jeu
HETEROGENEOUS_FREQUENCIES: Dict[str, Any] = {
    'production_industrielle': 'M',
    'inflation_ipc': 'M',
    'taux_chomage': 'M',
    'pib_trimestriel': 'Q',
    'balance_commerciale_annuelle': 'Y',
    'depenses_publiques_pib': {
        ('France',): 'Y', ('Allemagne',): 'Q', ('Italie',): 'Y',
    },
    'climat_affaires': 'M',
}

# Entités de heterogeneous_coverage_panel, dans l'ordre des données
HETEROGENEOUS_ENTITIES: Tuple[str, ...] = ('Allemagne', 'France', 'Italie')


def monthly_grid(start: str = '2021-01-31', periods: int = 36) -> pd.DatetimeIndex:
    """Build a month-end grid.

    Args:
        start: First month end.
        periods: Number of months.

    Returns:
        ``DatetimeIndex`` at ``'ME'``.
    """
    return pd.date_range(start=start, periods=periods, freq='ME')


def quarterly_grid(start: str = '2021-03-31', periods: int = 12) -> pd.DatetimeIndex:
    """Build a quarter-end grid.

    Args:
        start: First quarter end.
        periods: Number of quarters.

    Returns:
        ``DatetimeIndex`` at ``'QE'``.
    """
    return pd.date_range(start=start, periods=periods, freq='QE')


def annual_grid(start: str = '2021-12-31', periods: int = 3) -> pd.DatetimeIndex:
    """Build a year-end grid.

    Args:
        start: First year end.
        periods: Number of years.

    Returns:
        ``DatetimeIndex`` at ``'YE'``.
    """
    return pd.date_range(start=start, periods=periods, freq='YE')


def annual_anchor_grid() -> pd.DatetimeIndex:
    """Build the three annual anchors of the ``TS`` dataset, without frequency.

    Returns:
        ``DatetimeIndex`` 2021-12-31, 2022-12-31, 2023-12-31, a subset of the
        quarterly grid.
    """
    return pd.DatetimeIndex(pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31']))


def panel_grid(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.MultiIndex:
    """Pair the data's entities, in data order, with ``dates``.

    Args:
        data: Panel whose first index level holds the entities.
        dates: Dates given to every entity.

    Returns:
        Panel ``MultiIndex`` named like ``data.index``.
    """
    entities = data.index.get_level_values(0).unique().tolist()
    return pd.MultiIndex.from_product([entities, dates], names=list(data.index.names))


def entity_grid(
    entities: Sequence[str],
    dates: pd.DatetimeIndex,
    names: Sequence[str] = ('country', 'date'),
) -> pd.MultiIndex:
    """Pair explicitly ordered entities with ``dates``.

    Args:
        entities: Entities, in the order the grid must list them.
        dates: Dates given to every entity.
        names: Level names.

    Returns:
        Panel ``MultiIndex``.
    """
    return pd.MultiIndex.from_product([list(entities), dates], names=list(names))


def block_grid(
    blocks: Mapping[str, pd.DatetimeIndex],
    names: Sequence[str] = ('country', 'date'),
) -> pd.MultiIndex:
    """Build a panel grid where each entity keeps its own dates.

    Args:
        blocks: Mapping entity -> dates of that entity, in grid order.
        names: Level names.

    Returns:
        Panel ``MultiIndex`` concatenating the entity blocks.
    """
    tuples = [(entity, date) for entity, dates in blocks.items() for date in dates]
    return pd.MultiIndex.from_tuples(tuples, names=list(names))


def prime(
    materializer: CovariateMaterializer,
    column: str,
    values: pd.Series,
    freq: Optional[str],
    origin: str,
) -> None:
    """Record one production in the three stores, as an earlier stage would have.

    Goes through the public writer ``record_production``, so successive calls
    combine by ``combine_first`` like real stages.

    Args:
        materializer: Component whose stores are primed.
        column: Column produced.
        values: Values produced, indexed like a grid.
        freq: Production frequency of every cell.
        origin: ``CellOrigin`` of every cell.
    """
    materializer.record_production(
        column,
        values,
        pd.Series(origin, index=values.index, dtype='str'),
        pd.Series(freq, index=values.index, dtype='str'),
    )


def filled_dates(frame: pd.DataFrame, column: str, entity: Optional[str] = None) -> Set:
    """Return the dates where ``column`` is filled, for one entity.

    Args:
        frame: Feature frame, time series or panel.
        column: Column inspected.
        entity: Entity of a panel, None for a time series.

    Returns:
        Set of the dates holding a non-NaN value.
    """
    series = frame[column] if entity is None else frame.loc[entity, column]
    return set(series.dropna().index)


def origins_per_entity(store: pd.Series) -> Dict[Any, Set[str]]:
    """Group the origins of one panel store column by entity.

    Args:
        store: One column of ``origin_store``, panel-indexed.

    Returns:
        Mapping entity -> set of the origins written for it.
    """
    return {
        entity: set(block.dropna().unique())
        for entity, block in store.groupby(level=0)
    }


class SpyApplier:
    """Minimal double of ``AggregationConstraintApplier``: records, then shifts.

    The returned mask is deliberately inconsistent (all False): the
    materializer must not read it.

    Args:
        offset: Constant added to the values handed over, so that the
            features show whether the applier's output was kept.
    """

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: List[Dict[str, Any]] = []

    def rescale(
        self,
        values: pd.Series,
        observations: pd.Series,
        period_freq: str,
        column: Optional[str] = None,
        grid_freq: Any = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Record the call, then return the values shifted by ``offset``.

        Args:
            values: Values produced on the grid.
            observations: Totals the values should be rescaled to.
            period_freq: Frequency of the periods of the totals.
            column: Column being rescaled.
            grid_freq: Frequency of the grid of ``values``.

        Returns:
            Tuple ``(values + offset, all-False mask)``.
        """
        self.calls.append({
            'values': values.copy(),
            'observations': observations.copy(),
            'period_freq': period_freq,
            'column': column,
            'grid_freq': grid_freq,
        })
        return values + self.offset, pd.Series(False, index=values.index)
