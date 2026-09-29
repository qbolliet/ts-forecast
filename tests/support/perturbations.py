"""Toolbox of index and column perturbations for the tests.

Pure functions: each one takes a ``DataFrame`` (or ``Series``) already
built by :mod:`tests.support.datasets` or a fixture of
:mod:`tests.support.fixtures`, and returns a perturbed copy, never
modifying its input in place. They materialize the edge cases listed in
``CLAUDE.md`` ("test priorities": unsorted data, duplicated index,
missing entities, irregular frequencies, special column names, index
robustness) so that each component test composes them freely without
rewriting them.
"""
# Modules de base
from typing import Optional, Sequence, Union

import pandas as pd
from pandas.tseries.frequencies import get_period_alias

from tsforecast.utils.position import convert_position


def shuffle_rows(
    df: Union[pd.DataFrame, pd.Series], seed: int = 0
) -> Union[pd.DataFrame, pd.Series]:
    """Shuffle all rows of a time series or panel frame.

    Edge case: intentionally unsorted data, without touching values nor
    the (index, row) pairing.

    Args:
        df: Time series or panel data, any index type.
        seed: Seed of the pseudo-random permutation, for reproducibility.

    Returns:
        A new object with the same rows, in a shuffled order.

    Examples:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        >>> shuffled = shuffle_rows(s, seed=0)
        >>> sorted(shuffled.index) == sorted(s.index)
        True
    """
    # Cas limite (CLAUDE.md « Robustesse aux index ») : données intentionnellement
    # mal triées, sans toucher aux valeurs ni à l'appariement (index, ligne).
    return df.sample(frac=1.0, random_state=seed)


def reverse_entities(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Reverse the order of entities of a panel, keeping each entity's rows contiguous.

    Edge case: entities appearing neither in alphabetical order nor in the
    first-observation order a naive component expects.

    Args:
        df: Panel data with a ``MultiIndex`` whose first level is the entity.

    Returns:
        A new object with the same rows, entity blocks in reverse order.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('A', 0), ('A', 1), ('B', 0)], names=['entity', 'date']
        ... )
        >>> df = pd.DataFrame({'v': [1, 2, 3]}, index=idx)
        >>> list(reverse_entities(df).index.get_level_values('entity'))
        ['B', 'A', 'A']
    """
    # Cas limite (CLAUDE.md « Robustesse aux index ») : panel dont les entités
    # n'apparaissent pas dans l'ordre alphabétique ni dans l'ordre de première
    # observation attendu par un composant naïf — l'ordre au sein de chaque
    # entité reste inchangé, seul l'ordre des blocs d'entités est inversé.
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("reverse_entities requires a MultiIndex (panel data)")

    # Ordre de première apparition (et non trié) : l'inversion porte sur cet
    # ordre, pas sur l'ordre alphabétique, pour rester une pure permutation.
    entities = list(dict.fromkeys(df.index.get_level_values(0)))
    reversed_entities = list(reversed(entities))
    blocks = [df.loc[[entity]] for entity in reversed_entities]
    return pd.concat(blocks)


_SPECIAL_COLUMN_SUFFIXES = [
    ' (brute)',   # espace + parenthèses
    ' %',         # espace + pourcentage
    '/mois',      # slash
    ' à crédit',  # espace + accents
    ' (n°2)',     # parenthèse + caractère spécial supplémentaire
]


def with_special_column_names(
    df: Union[pd.DataFrame, pd.Series]
) -> tuple:
    """Rename every column with special characters (spaces, accents, ``/ % (``).

    Edge case: components building derived column names (string joins,
    regular expressions) are deemed fragile to these characters.

    Args:
        df: Time series or panel DataFrame (or Series, renamed via ``.name``).

    Returns:
        A tuple ``(renamed, mapping)`` where ``renamed`` is a copy of ``df``
        with special column names, and ``mapping`` associates each original
        column name to its special replacement.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1], 'b': [2]})
        >>> renamed, mapping = with_special_column_names(df)
        >>> mapping['a']
        'a (brute)'
        >>> list(renamed.columns) == list(mapping.values())
        True
    """
    # Cas limite (CLAUDE.md « noms de colonnes avec caractères spéciaux ») :
    # les composants qui construisent des noms de colonnes dérivés (jointures
    # de chaînes, expressions régulières) sont réputés fragiles à ces
    # caractères ; le mappage renvoyé permet de retrouver le nom d'origine.
    if isinstance(df, pd.Series):
        suffix = _SPECIAL_COLUMN_SUFFIXES[0]
        new_name = f"{df.name}{suffix}"
        mapping = {df.name: new_name}
        renamed = df.rename(new_name)
        return renamed, mapping

    mapping = {}
    new_columns = []
    for i, col in enumerate(df.columns):
        suffix = _SPECIAL_COLUMN_SUFFIXES[i % len(_SPECIAL_COLUMN_SUFFIXES)]
        new_col = f"{col}{suffix}"
        mapping[col] = new_col
        new_columns.append(new_col)

    renamed = df.copy()
    renamed.columns = new_columns
    return renamed, mapping


def with_index_names(
    df: Union[pd.DataFrame, pd.Series], names: Union[str, Sequence[Optional[str]]]
) -> Union[pd.DataFrame, pd.Series]:
    """Rename the index (or every level of a ``MultiIndex``) of a frame.

    Edge case: components locating the time level by a fixed name
    (``'date'``) rather than by position must fail on this renaming.

    Args:
        df: Time series or panel data.
        names: New name for a single index, or one name per level for a
            ``MultiIndex`` (same length and order as ``df.index.names``).

    Returns:
        A copy of ``df`` with the renamed index.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))
        >>> with_index_names(df, 'periode').index.name
        'periode'
    """
    # Cas limite (CLAUDE.md « noms de colonnes/index non standards ») : les
    # composants qui repèrent le niveau temporel par un nom fixe (``'date'``)
    # plutôt que par position doivent être mis en défaut par ce renommage.
    out = df.copy()
    out.index = out.index.set_names(names)
    return out


def to_three_level_index(
    df: Union[pd.DataFrame, pd.Series],
    region_by_entity: Optional[dict] = None,
    level_name: str = 'region',
    default_region: str = 'Zone euro',
) -> Union[pd.DataFrame, pd.Series]:
    """Add an outer entity level (``region``) to a two-level panel index.

    Edge case: components assuming a ``MultiIndex`` of exactly two levels
    (entity, date) misbehave once an extra level precedes the entity.

    Args:
        df: Panel data with a two-level ``MultiIndex`` (entity, date).
        region_by_entity: Mapping from entity name to region name. ``None``
            (default) assigns ``default_region`` to every entity — a constant
            outer level, sufficient to exercise ``nlevels == 3`` without
            changing the entity/region relationship.
        level_name: Name of the new outer level.
        default_region: Region used for every entity when ``region_by_entity``
            is ``None``, or for an entity missing from ``region_by_entity``.

    Returns:
        A copy of ``df`` with a three-level ``MultiIndex``
        (``level_name``, entity, date).

    Raises:
        TypeError: If ``df`` does not have a two-level ``MultiIndex``.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples([('FR', 0), ('DE', 0)], names=['country', 'date'])
        >>> df = pd.DataFrame({'v': [1, 2]}, index=idx)
        >>> to_three_level_index(df).index.names
        FrozenList(['region', 'country', 'date'])
    """
    # Cas limite (CLAUDE.md « index mixtes ») : composants qui supposent un
    # ``MultiIndex`` à exactement deux niveaux (entité, date) et se comportent
    # mal (mauvais niveau pris pour la date, groupby erroné) dès qu'un niveau
    # supplémentaire est ajouté avant l'entité.
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise TypeError("to_three_level_index requires a two-level MultiIndex (entity, date)")

    entities = df.index.get_level_values(0)
    if region_by_entity is None:
        regions = pd.Index([default_region] * len(df))
    else:
        regions = entities.map(lambda e: region_by_entity.get(e, default_region))

    new_index = pd.MultiIndex.from_arrays(
        [regions, entities, df.index.get_level_values(1)],
        names=[level_name] + list(df.index.names),
    )
    out = df.copy()
    out.index = new_index
    return out


def _infer_current_position(index: pd.Index) -> Optional[str]:
    """Infer whether a ``DatetimeIndex`` looks start- or end-anchored.

    An index is deemed "period start" if its first date falls on the 1st
    of the month (``MS`` / ``QS`` / ``YS`` convention of the whole
    package), "period end" otherwise. ``None`` for an empty index
    (nothing to infer).
    """
    if len(index) == 0:
        return None
    first_date = index[0]
    return 'S' if first_date.day == 1 else 'E'


def to_period_start(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Flip a start/end-anchored index to start-of-period anchoring.

    Edge case: components only tested under the native convention of the
    reference datasets (``MS`` for notebooks 2/3, ``ME`` for ``PANEL-X``)
    must stay correct under the other convention. The position is a
    property of the shared index, not of a single column; the frequency is
    applied per entity by :func:`convert_position`.

    Args:
        df: Time series or panel data with a ``DatetimeIndex`` (last level for
            a panel), already anchored at period start or period end.

    Returns:
        A copy of ``df`` anchored at period start ; unchanged (as a copy) if
        already start-anchored, if the index is empty, or in the rare case
        where :func:`convert_position` cannot infer any frequency at all
        (documented as a no-op rather than raised). A merely irregular index
        (some entity-specific anchors outside the common grid, as in
        :func:`~tests.support.fixtures.irregular_index_timeseries`) is usually still
        converted: :func:`convert_position` only needs a locally detectable
        step between dates, not a fully regular grid.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-31', periods=2, freq='ME')
        >>> df = pd.DataFrame({'v': [1, 2]}, index=dates)
        >>> to_period_start(df).index[0]
        Timestamp('2020-01-01 00:00:00')
    """
    # Cas limite (CLAUDE.md « positions début (MS, QS, YS) et fin (ME, QE, YE) »)
    # : composants qui ne testent que la convention native des jeux de
    # référence (``MS`` pour les notebooks 2/3, ``ME`` pour ``PANEL-X``)
    # doivent rester corrects sous l'autre convention.
    #
    # Règle : la position (début/fin) est une propriété de l'INDEX PARTAGÉ, pas
    # d'une colonne individuelle — toutes les colonnes d'une même ligne portent
    # la même date physique. La fréquence propre à chaque couple (entité,
    # colonne) n'intervient qu'indirectement, via :func:`convert_position`, qui
    # (pour un panel) détecte et applique la fréquence de CHAQUE ENTITÉ
    # séparément (``_convert_panel``) plutôt qu'une fréquence globale unique.
    dates = df.index.get_level_values(-1) if isinstance(df.index, pd.MultiIndex) else df.index
    current_position = _infer_current_position(dates)
    if current_position in (None, 'S'):
        return df.copy()
    try:
        return convert_position(df, current_position, 'start')
    except ValueError:
        # Fréquence non détectable (index trop irrégulier) : aucune conversion
        # possible, la perturbation se réduit à un no-op documenté.
        return df.copy()


def to_period_end(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Flip a start/end-anchored index to end-of-period anchoring.

    Same rule as :func:`to_period_start`, the opposite direction.

    Args:
        df: Time series or panel data with a ``DatetimeIndex`` (last level for
            a panel), already anchored at period start or period end.

    Returns:
        A copy of ``df`` anchored at period end ; unchanged (as a copy) if
        already end-anchored, or if no frequency could be inferred.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=2, freq='MS')
        >>> df = pd.DataFrame({'v': [1, 2]}, index=dates)
        >>> to_period_end(df).index[0].month
        1
    """
    dates = df.index.get_level_values(-1) if isinstance(df.index, pd.MultiIndex) else df.index
    current_position = _infer_current_position(dates)
    if current_position in (None, 'E'):
        return df.copy()
    try:
        return convert_position(df, current_position, 'end')
    except ValueError:
        return df.copy()


def to_period_index(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Convert the ``DatetimeIndex`` (last level for a panel) to a ``PeriodIndex``.

    Edge case: components calling ``DatetimeIndex``-specific methods
    (``.freq``, ``DateOffset`` arithmetic) without going through the
    package converters must fail on a ``PeriodIndex``.

    Args:
        df: Time series or panel data with a ``DatetimeIndex``.

    Returns:
        A copy of ``df`` with the date level converted to ``PeriodIndex``,
        left unchanged (where relevant) when the frequency cannot be
        inferred from an irregular index.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2]}, index=pd.date_range('2020-01-01', periods=2, freq='MS'))
        >>> isinstance(to_period_index(df).index, pd.PeriodIndex)
        True
    """
    # Cas limite (CLAUDE.md « types de dates : datetime64, Period, Timestamp »)
    # : composants qui appellent des méthodes propres à ``DatetimeIndex``
    # (``.freq``, arithmétique par ``DateOffset``) sans passer par les
    # convertisseurs du paquet doivent être mis en défaut par un ``PeriodIndex``.
    def _as_period(index: pd.DatetimeIndex) -> Optional[pd.PeriodIndex]:
        freq = index.freq
        if freq is None:
            # ``pd.infer_freq`` exige une séquence triée et sans répétition :
            # sur le niveau date d'un panel, les mêmes dates reviennent une
            # fois par entité et ne sont pas globalement monotones. On infère
            # donc sur les dates uniques triées, puis on l'applique à l'index
            # d'origine (non trié, avec répétitions).
            unique_sorted = pd.DatetimeIndex(sorted(index.unique()))
            try:
                freq = pd.infer_freq(unique_sorted)
            except ValueError:
                # Moins de 3 dates distinctes : pandas ne tente même pas
                # l'inférence (cas des jeux réduits à une poignée de lignes).
                freq = None
        if freq is None:
            return None
        # ``PeriodIndex`` ne connaît pas la position (« MS » / « QS » / « YS »
        # début, « ME » / « QE » / « YE » fin) : seule la granularité compte
        # (« M » / « Q » / « Y »). ``get_period_alias`` fait cette conversion ;
        # sans elle, ``to_period('MS')`` lève (alias non supporté en période).
        freq = get_period_alias(freq) or freq
        return index.to_period(freq)

    if isinstance(df.index, pd.MultiIndex):
        level = df.index.nlevels - 1
        dates = df.index.get_level_values(level)
        period_dates = _as_period(dates)
        if period_dates is None:
            return df.copy()
        arrays = [df.index.get_level_values(i) for i in range(level)] + [period_dates]
        out = df.copy()
        out.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
        return out

    period_dates = _as_period(df.index)
    if period_dates is None:
        return df.copy()
    out = df.copy()
    out.index = period_dates
    out.index.name = df.index.name
    return out


def drop_entity(
    df: Union[pd.DataFrame, pd.Series], entity: str
) -> Union[pd.DataFrame, pd.Series]:
    """Remove every row belonging to one entity of a panel.

    Edge case: an entity absent from the dataset (neither observed nor
    present as NaN), to be distinguished from an entity present but
    entirely NaN.

    Args:
        df: Panel data with a ``MultiIndex`` whose first level is the entity.
        entity: Name of the entity to remove.

    Returns:
        A copy of ``df`` without the rows of ``entity``.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples([('A', 0), ('B', 0)], names=['entity', 'date'])
        >>> df = pd.DataFrame({'v': [1, 2]}, index=idx)
        >>> list(drop_entity(df, 'A').index.get_level_values('entity'))
        ['B']
    """
    # Cas limite (CLAUDE.md « entités manquantes dans les panels ») : simule
    # une entité absente du jeu (ni observée ni présente en NaN), à distinguer
    # d'une entité présente mais entièrement NaN.
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("drop_entity requires a MultiIndex (panel data)")
    return df.drop(index=entity, level=0)


def with_duplicated_rows(
    df: Union[pd.DataFrame, pd.Series], n: int = 1
) -> Union[pd.DataFrame, pd.Series]:
    """Duplicate the first ``n`` rows by appending them a second time.

    Edge case: duplicates are appended at the end of the frame (unsorted),
    to exercise robustness to disorder along with duplication.

    Args:
        df: Time series or panel data.
        n: Number of leading rows to duplicate.

    Returns:
        A copy of ``df`` with its first ``n`` rows appearing twice.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2, 3]}, index=[0, 1, 2])
        >>> len(with_duplicated_rows(df, n=1))
        4
    """
    # Cas limite (CLAUDE.md « index dupliqués ») : les doublons sont ajoutés en
    # queue de frame (pas triés), pour aussi éprouver la robustesse au
    # désordonnancement en même temps que la duplication.
    duplicated = df.iloc[:n]
    return pd.concat([df, duplicated])


def single_observation(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Reduce a frame to its single first row, keeping the original index type.

    Edge case: no frequency can be inferred from a single point;
    components blindly calling ``infer_freq`` must tolerate it.

    Args:
        df: Time series or panel data with at least one row.

    Returns:
        A copy of ``df`` restricted to its first row.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2, 3]}, index=[0, 1, 2])
        >>> len(single_observation(df))
        1
    """
    # Cas limite (CLAUDE.md « datasets ... avec une seule observation ») :
    # aucune fréquence n'est inférable depuis un seul point, les composants
    # qui appellent aveuglément ``infer_freq`` doivent le tolérer.
    return df.iloc[[0]].copy()


def empty_like(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Reduce a frame to zero rows, keeping its columns, dtypes and index type.

    Edge case: distinguishes an empty dataset from a missing one (``None``):
    the shape (columns, dtypes, index names) stays fully defined, only the
    rows disappear.

    Args:
        df: Time series or panel data.

    Returns:
        A copy of ``df`` with zero rows.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2, 3]}, index=[0, 1, 2])
        >>> len(empty_like(df))
        0
        >>> list(empty_like(df).columns)
        ['v']
    """
    # Cas limite (CLAUDE.md « datasets vides ») : distingue un jeu vide d'un
    # jeu absent (``None``) — la forme (colonnes, dtypes, noms d'index) reste
    # entièrement renseignée, seules les lignes disparaissent.
    return df.iloc[0:0].copy()
