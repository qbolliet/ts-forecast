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

import numpy as np
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


def to_period_start_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Re-date every observation at the start of ITS OWN period.

    Edge case: the start-anchored counterpart of a mixed-frequency frame.
    :func:`to_period_start` moves the shared index, so the yearly anchor of a
    monthly grid lands on December 1st — the start of December, not of the
    year. Here each (entity, column) pair is read at its own detected
    frequency, and each of its observations moves to the first day of the
    period it covers: a yearly value to January 1st, a quarterly one to the
    first day of its quarter, a monthly one to the first of its month. The
    grid itself is :func:`to_period_start` of the index.

    Args:
        df: Time series or panel ``DataFrame`` (``MultiIndex`` with the date
            last), anchored at period end, on a grid fine enough to hold every
            period start (monthly for monthly, quarterly and yearly columns).

    Returns:
        A copy on the start-anchored grid, every observation re-dated, NaN
        elsewhere.

    Raises:
        ValueError: If the frequency of an observed (entity, column) pair
            cannot be detected (fewer than two observations), or if a period
            start falls outside the start-anchored grid.

    Examples:
        >>> import numpy as np
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> yearly = [np.nan] * 11 + [120.0] + [np.nan] * 11 + [132.0]
        >>> df = pd.DataFrame({'m': np.arange(24.0), 'y': yearly}, index=dates)
        >>> start = to_period_start_by_column(df)
        >>> start['y'].dropna().index.strftime('%Y-%m-%d').tolist()
        ['2021-01-01', '2022-01-01']
        >>> start['m'].tolist() == df['m'].tolist()
        True
    """
    # Détection par couple (entité, colonne) : importée ici pour garder le
    # module léger à l'import
    from tsforecast.utils.frequency import detect_frequency, normalize_frequency

    # Grille de début de période, et trame vide à remplir
    grid = to_period_start(df).index
    out = pd.DataFrame(float('nan'), index=grid, columns=df.columns)

    # Blocs d'entité : un seul bloc d'entité vide pour une série temporelle
    is_panel = isinstance(df.index, pd.MultiIndex)
    if is_panel:
        # Niveau scalaire pour une seule entité : pandas rendra sinon des tuples
        levels = list(range(df.index.nlevels - 1))
        level = levels[0] if len(levels) == 1 else levels
        blocks = [
            (entity if isinstance(entity, tuple) else (entity,), block)
            for entity, block in df.groupby(level=level, sort=False)
        ]
    else:
        blocks = [((), df)]

    # Chaque observation est redatée au début de SA période
    for entity, block in blocks:
        dates = block.index.get_level_values(-1)
        for column in df.columns:
            observed = pd.Series(block[column].to_numpy(), index=dates).dropna()
            if observed.empty:
                continue
            try:
                frequency = detect_frequency(data=observed)
            except ValueError as error:
                raise ValueError(
                    f"Cannot detect the frequency of {entity + (column,)}: {error}"
                ) from error
            if frequency is None:
                raise ValueError(f"Cannot detect the frequency of {entity + (column,)}")
            base = normalize_frequency(frequency, return_format='base')
            starts = observed.index.to_period(base).start_time
            keys = [(*entity, start) for start in starts] if is_panel else list(starts)
            outside = [key for key in keys if key not in out.index]
            if outside:
                raise ValueError(
                    f"Period starts outside the start-anchored grid for "
                    f"{entity + (column,)}: {outside[:3]}"
                )
            out.loc[keys, column] = observed.to_numpy()
    return out


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
        # (« M » / « Q » / « Y »). ``get_period_alias`` fait cette conversion,
        # sur l'alias en chaîne (un objet offset n'est pas converti) ; sans elle,
        # ``to_period('MS')`` lève (alias non supporté en période).
        alias = freq if isinstance(freq, str) else freq.freqstr
        return index.to_period(get_period_alias(alias) or alias)

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


def _as_entity_tuple(key: object) -> tuple:
    """Return an entity key as a tuple (a scalar becomes a one-element tuple)."""
    return key if isinstance(key, tuple) else (key,)


def _entity_row_mask(df: Union[pd.DataFrame, pd.Series], entity: object) -> np.ndarray:
    """Boolean mask of the rows whose entity levels equal ``entity``.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.
        ValueError: If the number of labels of ``entity`` differs from the
            number of entity levels (all levels but the last).
        KeyError: If no row belongs to ``entity``.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("a MultiIndex (panel data) is required")
    key = _as_entity_tuple(entity)
    n_levels = df.index.nlevels - 1
    if len(key) != n_levels:
        raise ValueError(f"entity {key!r} has {len(key)} labels, expected {n_levels}")
    mask = np.ones(len(df), dtype=bool)
    for level, label in enumerate(key):
        mask &= np.asarray(df.index.get_level_values(level) == label)
    if not mask.any():
        raise KeyError(entity)
    return mask


def rename_entities(
    df: Union[pd.DataFrame, pd.Series], mapping: dict
) -> Union[pd.DataFrame, pd.Series]:
    """Relabel the entities of a panel, keeping every row at its place.

    Edge case: entity keys that are not plain ASCII strings — labels with
    spaces or accents, integers, multi-level keys — must travel through the
    components unchanged (masks, dictionaries keyed by entity, rebuilt
    indexes).

    Args:
        df: Panel data with a ``MultiIndex`` (entity levels first, time last).
        mapping: Old entity key -> new entity key. A key is a scalar for a
            panel with a single entity level, a tuple (one label per entity
            level) otherwise; a value has the same shape as its key. Entities
            absent from the mapping keep their labels.

    Returns:
        A copy of ``df`` whose entity levels are relabelled, same row order,
        same level names.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.
        KeyError: If a key of ``mapping`` matches no entity of ``df``.
        ValueError: If a key or a value does not have one label per entity
            level, or if two entities would receive the same new key.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('FR', 0), ('DE', 0)], names=['entity', 'date']
        ... )
        >>> df = pd.DataFrame({'v': [1, 2]}, index=idx)
        >>> list(rename_entities(df, {'FR': 'Côte d\\'Ivoire'}).index.get_level_values(0))
        ["Côte d'Ivoire", 'DE']
    """
    # Cas limite : clés d'entité qui ne sont pas de simples chaînes ASCII
    # (espaces, accents, entiers, clés multi-niveaux).
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("rename_entities requires a MultiIndex (panel data)")
    n_levels = df.index.nlevels - 1
    renames = {}
    for old, new in mapping.items():
        old_key, new_key = _as_entity_tuple(old), _as_entity_tuple(new)
        if len(new_key) != n_levels:
            raise ValueError(f"new key {new!r} has {len(new_key)} labels, expected {n_levels}")
        # Contrôle de l'existence (et de l'arité) de la clé d'origine
        _entity_row_mask(df, old_key)
        renames[old_key] = new_key

    # Clé d'entité de chaque ligne, puis relecture niveau par niveau
    entity_rows = list(zip(*(df.index.get_level_values(i) for i in range(n_levels))))
    new_rows = [renames.get(row, row) for row in entity_rows]
    # Deux entités distinctes ne doivent pas fusionner
    before = dict.fromkeys(entity_rows)
    if len({renames.get(key, key) for key in before}) != len(before):
        raise ValueError("two entities would receive the same new key")

    arrays = [pd.Index([row[i] for row in new_rows]) for i in range(n_levels)]
    arrays.append(df.index.get_level_values(-1))
    out = df.copy()
    out.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
    return out


def interleave_entities(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Order the rows of a panel by date first, then by entity.

    Edge case: entity rows that are not contiguous, so that a component
    assuming one block per entity (``groupby`` on consecutive rows, slicing
    by run) reads the wrong rows. Entities keep their order of first
    appearance within a date; the order of the rows of an entity at a given
    date is kept.

    Args:
        df: Panel data with a ``MultiIndex`` (entity levels first, time last).

    Returns:
        A copy of ``df`` sorted by date, then by order of first appearance of
        the entity.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_product([['B', 'A'], [0, 1]], names=['entity', 'date'])
        >>> df = pd.DataFrame({'v': [1, 2, 3, 4]}, index=idx)
        >>> list(interleave_entities(df).index)
        [('B', 0), ('A', 0), ('B', 1), ('A', 1)]
    """
    # Cas limite : lignes d'une même entité non contiguës (tri date puis entité)
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("interleave_entities requires a MultiIndex (panel data)")
    n_levels = df.index.nlevels - 1
    entity_rows = list(zip(*(df.index.get_level_values(i) for i in range(n_levels))))
    # Rang d'apparition de l'entité, rang chronologique de la date
    entity_rank = pd.factorize(pd.Series(entity_rows, dtype=object))[0]
    date_rank = pd.factorize(df.index.get_level_values(-1), sort=True)[0]
    # ``lexsort`` trie d'abord sur la dernière clé et est stable
    order = np.lexsort((entity_rank, date_rank))
    return df.iloc[order].copy()


def unsort_within_entities(
    df: Union[pd.DataFrame, pd.Series], seed: int = 0
) -> Union[pd.DataFrame, pd.Series]:
    """Shuffle the dates inside each entity while keeping entity blocks contiguous.

    Edge case: a panel sorted by entity but not by date, as left by a
    concatenation of per-entity extracts. Entity blocks keep their order of
    first appearance. Every entity of at least two rows is guaranteed to
    change order.

    Args:
        df: Panel data with a ``MultiIndex`` (entity levels first, time last).
        seed: Seed of the pseudo-random permutations, for reproducibility.

    Returns:
        A copy of ``df`` with the same rows; each entity block is contiguous
        and its rows are permuted.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], range(3)], names=['entity', 'date'])
        >>> df = pd.DataFrame({'v': range(6)}, index=idx)
        >>> out = unsort_within_entities(df, seed=0)
        >>> list(out.index.get_level_values('entity'))
        ['A', 'A', 'A', 'B', 'B', 'B']
        >>> out.index.get_level_values('date')[:3].is_monotonic_increasing
        False
    """
    # Cas limite : panel trié par entité mais pas par date
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("unsort_within_entities requires a MultiIndex (panel data)")
    n_levels = df.index.nlevels - 1
    entity_rows = list(zip(*(df.index.get_level_values(i) for i in range(n_levels))))
    entity_rank = pd.factorize(pd.Series(entity_rows, dtype=object))[0]
    rng = np.random.default_rng(seed)
    positions = []
    for rank in range(entity_rank.max() + 1 if len(entity_rank) else 0):
        rows = np.flatnonzero(entity_rank == rank)
        permutation = rng.permutation(len(rows))
        # Une permutation identité laisserait l'entité triée : on l'inverse
        if len(rows) > 1 and (permutation == np.arange(len(rows))).all():
            permutation = permutation[::-1]
        positions.append(rows[permutation])
    order = np.concatenate(positions) if positions else np.array([], dtype=int)
    return df.iloc[order].copy()


def with_single_observation_entity(
    df: Union[pd.DataFrame, pd.Series], entity: object
) -> Union[pd.DataFrame, pd.Series]:
    """Keep a single row (the first one) for one entity of a panel.

    Edge case: an entity too short to carry any frequency, any split or any
    window, next to entities with a normal history.

    Args:
        df: Panel data with a ``MultiIndex`` (entity levels first, time last).
        entity: Entity key to reduce: a scalar for a single entity level, a
            tuple otherwise.

    Returns:
        A copy of ``df`` where ``entity`` keeps only its first row; the other
        entities and the row order are unchanged.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.
        ValueError: If ``entity`` does not have one label per entity level.
        KeyError: If ``entity`` matches no row.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], range(3)], names=['entity', 'date'])
        >>> df = pd.DataFrame({'v': range(6)}, index=idx)
        >>> list(with_single_observation_entity(df, 'A').index)
        [('A', 0), ('B', 0), ('B', 1), ('B', 2)]
    """
    # Cas limite : entité réduite à une observation, à côté d'entités normales
    mask = _entity_row_mask(df, entity)
    # Première ligne de l'entité, dans l'ordre des données
    first_row = np.flatnonzero(mask)[0]
    keep = ~mask
    keep[first_row] = True
    return df[keep].copy()


def truncate_entity(
    df: Union[pd.DataFrame, pd.Series], entity: object, end: object
) -> Union[pd.DataFrame, pd.Series]:
    """Drop the rows of one entity dated after ``end``.

    Edge case: an entity whose history stops before the others, hence absent
    from the last folds of a time split or from the last windows of a
    transformation.

    Args:
        df: Panel data with a ``MultiIndex`` (entity levels first, time last).
        entity: Entity key to truncate: a scalar for a single entity level, a
            tuple otherwise.
        end: Last date kept for ``entity`` (inclusive); anything accepted by
            ``pd.Timestamp``, or by ``pd.Period`` for a ``PeriodIndex``.

    Returns:
        A copy of ``df`` without the rows of ``entity`` dated after ``end``;
        the other entities and the row order are unchanged.

    Raises:
        TypeError: If ``df`` does not have a ``MultiIndex``.
        ValueError: If ``entity`` does not have one label per entity level.
        KeyError: If ``entity`` matches no row.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        >>> df = pd.DataFrame({'v': range(6)}, index=idx)
        >>> len(truncate_entity(df, 'A', '2020-02-01'))
        5
    """
    # Cas limite : entité dont l'historique s'arrête avant celui des autres
    mask = _entity_row_mask(df, entity)
    dates = df.index.get_level_values(-1)
    # Borne exprimée dans le type des dates du niveau temporel
    if isinstance(dates, pd.PeriodIndex):
        limit = pd.Period(end, freq=dates.freq)
    elif isinstance(dates, pd.DatetimeIndex):
        limit = pd.Timestamp(end)
    else:
        limit = end
    keep = ~(mask & np.asarray(dates > limit))
    return df[keep].copy()


# Résolutions de dates prises en charge par pandas, de la plus grossière à la plus fine
DATETIME_UNITS: tuple = ('s', 'ms', 'us', 'ns')


def _as_resolution(values: pd.Index, unit: str) -> pd.Index:
    """Cast a ``DatetimeIndex`` / ``TimedeltaIndex`` to ``unit``, other indexes unchanged.

    Raises:
        ValueError: If a value has a part finer than ``unit``.
    """
    if not isinstance(values, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return values
    converted = values.as_unit(unit)
    # Contrôle d'exactitude : ``as_unit`` vers une unité plus grossière tronque en silence
    if not (converted.as_unit(values.unit) == values)[~values.isna()].all():
        raise ValueError(f"Some dates are not representable exactly at resolution '{unit}'")
    return converted


def with_datetime_resolution(
    df: Union[pd.DataFrame, pd.Series], unit: str
) -> Union[pd.DataFrame, pd.Series]:
    """Store every date (index levels, datetime and timedelta columns) at one resolution.

    Edge case: the same instants stored in seconds, milliseconds, microseconds
    or nanoseconds. pandas infers the resolution from the input (a string or a
    ``datetime`` gives microseconds, a ``np.datetime64`` keeps its own unit),
    so a component must give the same result whatever the resolution of its
    input.

    Args:
        df: Time series or panel data: ``DatetimeIndex`` / ``TimedeltaIndex``
            index or levels, ``datetime64`` / ``timedelta64`` columns (time
            zone kept).
        unit: Target resolution, one of :data:`DATETIME_UNITS`.

    Returns:
        A copy of ``df`` whose dates have resolution ``unit``; the other
        levels, columns and values are unchanged.

    Raises:
        ValueError: If ``unit`` is not one of :data:`DATETIME_UNITS`, or if a
            date has a part finer than ``unit``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'v': [1, 2]}, index=pd.date_range('2020-01-01', periods=2, freq='MS'))
        >>> converted = with_datetime_resolution(df, 's')
        >>> str(converted.index.dtype)
        'datetime64[s]'
        >>> bool((converted.index == df.index).all())
        True
    """
    if unit not in DATETIME_UNITS:
        raise ValueError(f"unit must be one of {DATETIME_UNITS}, got {unit!r}")

    out = df.copy()
    # Index : chaque niveau de dates (ou de durées) converti, les autres inchangés
    if isinstance(out.index, pd.MultiIndex):
        arrays = [_as_resolution(out.index.get_level_values(i), unit) for i in range(out.index.nlevels)]
        out.index = pd.MultiIndex.from_arrays(arrays, names=out.index.names)
    else:
        # Index simple : ``as_unit`` conserve le nom et la fréquence de l'index
        out.index = _as_resolution(out.index, unit)

    # Colonnes de dates ou de durées, valeurs manquantes comprises
    columns = [out] if isinstance(out, pd.Series) else [out[col] for col in out.columns]
    for position, column in enumerate(columns):
        if not (pd.api.types.is_datetime64_any_dtype(column.dtype)
                or pd.api.types.is_timedelta64_dtype(column.dtype)):
            continue
        converted = _as_resolution(pd.Index(column), unit)
        if isinstance(out, pd.Series):
            out = pd.Series(converted, index=out.index, name=out.name)
        else:
            out.isetitem(position, converted)
    return out


def datetime_units(obj: Union[pd.DataFrame, pd.Series, pd.Index]) -> dict:
    """Map every date or duration location of an object to its resolution.

    Companion of :func:`with_datetime_resolution`: tells at which resolution a
    component returns its dates.

    Args:
        obj: ``DataFrame``, ``Series`` or ``Index``.

    Returns:
        ``{('index', level): unit}`` for each ``DatetimeIndex`` /
        ``TimedeltaIndex`` level (``level`` is the position), and
        ``{('column', name): unit}`` for each ``datetime64`` / ``timedelta64``
        column (a ``Series`` counts as one column).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'when': pd.to_datetime(['2024-01-01']).as_unit('s')},
        ...                   index=pd.date_range('2020-01-01', periods=1).as_unit('ms'))
        >>> datetime_units(df)
        {('index', 0): 'ms', ('column', 'when'): 's'}
    """
    units = {}
    index = obj if isinstance(obj, pd.Index) else obj.index
    levels = (
        [index.get_level_values(i) for i in range(index.nlevels)]
        if isinstance(index, pd.MultiIndex) else [index]
    )
    for position, level in enumerate(levels):
        if isinstance(level, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            units[('index', position)] = level.unit
    if isinstance(obj, pd.Index):
        return units
    columns = [obj] if isinstance(obj, pd.Series) else [obj[col] for col in obj.columns]
    for column in columns:
        if (pd.api.types.is_datetime64_any_dtype(column.dtype)
                or pd.api.types.is_timedelta64_dtype(column.dtype)):
            units[('column', column.name)] = column.dt.unit
    return units


def string_dtypes(obj: Union[pd.DataFrame, pd.Series, pd.Index]) -> dict:
    """Map every string or object location of an object to ``'str'`` or ``'object'``.

    Companion of :func:`datetime_units` for the string dtypes: tells whether a
    component returns its text in the ``str`` dtype (whatever its storage,
    python or pyarrow) or as ``object``. Locations of any other dtype are left out.

    Args:
        obj: ``DataFrame``, ``Series`` or ``Index``.

    Returns:
        ``{('index', level): kind}`` for each string or object index level
        (``level`` is the position), and ``{('column', name): kind}`` for each
        string or object column (a ``Series`` counts as one column).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'unit': ['day'], 'n': [1], 'tag': pd.Series([None], dtype=object)},
        ...                   index=pd.Index(['GDP'], name='column'))
        >>> string_dtypes(df)
        {('index', 0): 'str', ('column', 'unit'): 'str', ('column', 'tag'): 'object'}
    """
    def kind(dtype) -> Optional[str]:
        if isinstance(dtype, pd.StringDtype):
            return 'str'
        return 'object' if dtype == object else None

    kinds = {}
    index = obj if isinstance(obj, pd.Index) else obj.index
    levels = (
        [index.get_level_values(i) for i in range(index.nlevels)]
        if isinstance(index, pd.MultiIndex) else [index]
    )
    for position, level in enumerate(levels):
        if kind(level.dtype) is not None:
            kinds[('index', position)] = kind(level.dtype)
    if isinstance(obj, pd.Index):
        return kinds
    columns = [obj] if isinstance(obj, pd.Series) else [obj.iloc[:, i] for i in range(obj.shape[1])]
    for column in columns:
        if kind(column.dtype) is not None:
            kinds[('column', column.name)] = kind(column.dtype)
    return kinds
