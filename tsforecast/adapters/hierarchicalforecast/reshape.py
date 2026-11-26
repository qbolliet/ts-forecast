"""Conversion between wide hierarchical data and the long layout of ``HierarchicalForecastAdapter``.

``stack_nodes`` turns a frame with one column per hierarchy node (``value_a``,
``value_b``... for each date, or each entity and date) into the long layout read
by the adapter: one row per node, the node carried by an index level placed just
before the date. ``unstack_nodes`` is the inverse conversion.
"""
# Importation des modules
# Modules de base
from collections.abc import Hashable
from typing import Union

# Modules de manipulation de données
import pandas as pd


# Fonction de passage du format large au format long
def stack_nodes(data: pd.DataFrame, node_level: Hashable = "node") -> Union[pd.Series, pd.DataFrame]:
    """Move the hierarchy nodes from the columns of a wide frame to an index level.

    The node level is inserted just before the date, which stays the last
    index level. Empty cells are kept as missing values.

    Args:
        data: Wide frame indexed by a ``DatetimeIndex`` or an
            ``(entity..., date)`` ``MultiIndex``. Its columns are either the
            node labels, or a two-level ``MultiIndex`` ``(variable, node)``
            (several variables per node, for instance the forecasts of
            several base models).
        node_level: Name of the index level carrying the node labels.

    Returns:
        A ``Series`` (columns = nodes) or a ``DataFrame`` with one column per
        variable (columns = ``(variable, node)``), indexed by
        ``(entity..., node, date)``.

    Raises:
        TypeError: If ``data`` is not a ``DataFrame``.
        ValueError: If the columns have more than two levels, or if
            ``node_level`` already names an index level.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=2, freq='MS', name='date')
        >>> wide = pd.DataFrame({'a': [3.0, 5.0], 'b': [1.0, 2.0], 'c': [2.0, 3.0]}, index=dates)
        >>> long = stack_nodes(wide)
        >>> long.index.names, len(long)
        (FrozenList(['node', 'date']), 6)
        >>> long.loc['b'].tolist()
        [1.0, 2.0]
    """
    # Vérification de l'entrée
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    # Au plus deux niveaux de colonnes : (variable, nœud)
    if data.columns.nlevels > 2:
        raise ValueError(
            f"The columns must be the nodes or (variable, node) pairs, got {data.columns.nlevels} levels"
        )
    # Disponibilité du nom du niveau de nœuds
    if node_level in data.index.names:
        raise ValueError(f"node_level {node_level!r} already names an index level")

    # Niveau de colonnes des nœuds empilé en dernier niveau d'index, cellules vides conservées
    if isinstance(data.columns, pd.MultiIndex):
        # Colonnes (variable, nœud) : seul le dernier niveau est renommé
        columns = data.columns.set_names(node_level, level=-1)
    else:
        # Colonnes réduites aux nœuds
        columns = data.columns.rename(node_level)
    # Empilement des nœuds
    stacked = data.set_axis(columns, axis=1).stack(level=-1)
    # Niveau des nœuds replacé juste avant la date : il occupe après empilement la position
    # qui suit les niveaux de l'index d'entrée
    n_index = data.index.nlevels
    # Nouvel ordre des niveaux : entités, nœuds, date
    order = list(range(n_index - 1)) + [n_index, n_index - 1]
    return stacked.reorder_levels(order)


# Fonction de passage du format long au format large
def unstack_nodes(
    data: Union[pd.Series, pd.DataFrame], node_level: Hashable = "node"
) -> pd.DataFrame:
    """Move the hierarchy nodes from an index level to the columns.

    Inverse of :func:`stack_nodes`: node columns keep the order of first
    appearance of the nodes; rows are sorted by entity then date. A missing
    ``(entity, node, date)`` row becomes a missing value.

    Args:
        data: Long ``Series`` or ``DataFrame`` whose index has the level
            ``node_level``.
        node_level: Name or position of the index level carrying the node
            labels.

    Returns:
        Wide frame: one column per node for a ``Series``, ``(variable, node)``
        columns for a ``DataFrame``.

    Raises:
        TypeError: If ``data`` is neither a ``Series`` nor a ``DataFrame``.
        ValueError: If ``node_level`` is not an index level, or if it is the
            only index level.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=2, freq='MS', name='date')
        >>> wide = pd.DataFrame({'a': [3.0, 5.0], 'b': [1.0, 2.0]}, index=dates)
        >>> unstack_nodes(stack_nodes(wide)).equals(wide.rename_axis(columns='node'))
        True
    """
    # Vérification de l'entrée
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"data must be a pandas Series or DataFrame, got {type(data).__name__}")
    # Niveaux de nœuds et de date indispensables
    if data.index.nlevels < 2:
        raise ValueError("data needs a node level and a date level")
    # Vérification de la désignation du niveau : position hors noms de niveaux
    if isinstance(node_level, int) and node_level not in data.index.names:
        if not -data.index.nlevels <= node_level < data.index.nlevels:
            raise ValueError(f"node_level {node_level} is not an index level position")
    elif node_level not in data.index.names:
        # Cas d'un nom de niveau inexistant
        raise ValueError(f"{node_level!r} is not an index level; levels: {list(data.index.names)}")

    # Nœuds dans l'ordre de première apparition
    nodes = data.index.get_level_values(node_level).unique()
    # Passage du niveau de nœuds en colonnes
    wide = data.unstack(level=node_level)
    # Cas d'une série : une colonne par nœud
    if isinstance(data, pd.Series):
        return wide.reindex(columns=nodes)
    # Colonnes (variable, nœud) : variables dans l'ordre d'entrée, puis nœuds dans l'ordre d'apparition
    order = pd.MultiIndex.from_product([data.columns, nodes], names=wide.columns.names)
    return wide.reindex(columns=order)
