"""Cross-validators for panel data: global and per-entity variants.

``PanelOutOfSampleSplit`` and ``PanelInSampleSplit`` cut every entity at the
same test dates and yield one fold holding all entities; an entity that does
not hold the test dates of a fold only trains in it. The ``PerEntity``
variants cut each entity on its own dates and yield one fold per entity and
per split. Entities are read from the ``MultiIndex`` (every level but the
last, which holds the date) when ``groups`` is not given.
"""
# Importation des modules
# Modules de base
from collections.abc import Iterator
from typing import Any
import inspect
import warnings

import numpy as np
import pandas as pd

# Importations locales
from ..panel.utils import is_panel_data
from .base import (
    _DATES,
    InSampleSplit,
    OutOfSampleSplit,
    _entity_labels,
    _labels_array,
    _normalize_label,
    _parse_test_indices,
    _precompute_group_mappings,
    _resolve_test_positions,
    _time_level,
    _verify_and_sort_data,
)

# Sklearn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


# Fonction auxiliaire d'extraction des groupes depuis les données de panel
def _extract_groups_from_panel(X):
    """Extract the entity of each row of panel data.

    The entity is read from every index level but the last (the date): the
    value of the entity level when there is one, the tuple of the entity
    levels when there are several.

    Args:
        X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity..., time)

    Returns:
        np.ndarray: One-dimensional object array of the entity of each row

    Raises:
        ValueError: If X is not a pandas object with a MultiIndex of at least
            two levels

    Examples:
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2020-01-01', periods=2, freq='D')])
        >>> _extract_groups_from_panel(pd.DataFrame({'value': range(4)}, index=idx)).tolist()
        ['A', 'A', 'B', 'B']
        >>> idx = pd.MultiIndex.from_product([['FR'], ['a', 'b'], [2020]])
        >>> _extract_groups_from_panel(pd.Series(range(2), index=idx)).tolist()
        [('FR', 'a'), ('FR', 'b')]
    """
    # Vérification qu'il s'agit d'un objet pandas
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise ValueError("Panel data requires pandas DataFrame/Series with MultiIndex")

    # Vérification de la structure de panel : MultiIndex (entité..., date)
    if not is_panel_data(X):
        raise ValueError("Panel data requires MultiIndex (entity, time)")

    # Entité de chaque ligne : niveaux d'entité (tous sauf le dernier)
    return _entity_labels(X)


# Classe de base pour la validation croisée out-of-sample sur données de panel
class PanelOutOfSampleSplit(OutOfSampleSplit):
    """Out-of-sample cross-validator cutting every entity of a panel at common dates.

    The panel has a MultiIndex ``(entity..., date)``: the entity is made of
    every level but the last, which holds the date. Each fold tests the same
    dates for every entity, so that a model pooling the entities never
    trains on a date of the test period:

    - without ``test_indices``, ``n_splits`` adjacent blocks of ``test_size``
      dates (default 1) ending at the last date of the panel;
    - with dates, one fold per date, ``test_size`` dates from it.

    Each tested entity trains on its dates before the test, minus ``gap``,
    within ``max_train_size`` dates. An entity that does not hold the test
    date of a fold is warned about and not tested; it trains on its dates
    before that date (minus ``gap``). An integer position or an
    ``(entity..., date)`` tuple opens a fold that tests and trains its entity
    only. Positions are those of the rows of ``X`` as given; an unsorted
    panel is sorted internally (with a warning).

    Args:
        n_splits: Number of folds when ``test_indices`` is None (at least 2).
        test_indices: Dates (common to the entities), integer positions or
            ``(entity..., date)`` tuples; see ``OutOfSampleSplit``.
        max_train_size: Maximum number of training dates per entity, or
            None for an expanding window.
        test_size: Number of test dates per entity and per fold.
        gap: Number of dates skipped between training and test.

    Examples:
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        >>> index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'v': range(8)}, index=index)
        >>> [(train.tolist(), test.tolist()) for train, test in PanelOutOfSampleSplit(n_splits=2).split(X)]
        [([0, 1, 4, 5], [2, 6]), ([0, 1, 2, 4, 5, 6], [3, 7])]

        ``B`` ends in March: it is not tested in April, but trains on January to March.

        >>> X_short = X.drop(('B', dates[3]))
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore')
        ...     folds = [(train.tolist(), test.tolist()) for train, test in PanelOutOfSampleSplit(n_splits=2).split(X_short)]
        >>> folds
        [([0, 1, 4, 5], [2, 6]), ([0, 1, 2, 4, 5, 6], [3])]
    """

    # Méthode principale de séparation des données de panel
    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate the ``(train, test)`` positions of each fold.

        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity..., date).
            y (array-like, optional): Target; a pandas target must have the index of ``X``.
            groups (array-like, optional): Entity of each row; must repeat the
                entity levels of ``X``. Deduced from the index when None.

        Returns:
            Generator of ``(train_positions, test_positions)`` integer arrays.

        Raises:
            ValueError: If ``X`` is not panel data, or for any error of
                ``OutOfSampleSplit.split`` (invalid parameter, repeated
                entity-date pair, inconsistent ``groups``, unknown test date,
                too many splits, tested entity without training data).
        """
        # Extraction des groupes depuis la structure du panel si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)

        # Appel de la méthode parente avec les groupes pour le comportement panel
        return super().split(X, y, groups)


# Classe de base pour la validation croisée in-sample sur données de panel
class PanelInSampleSplit(InSampleSplit):
    """In-sample cross-validator testing every entity of a panel at common dates.

    It yields a single fold whose training set contains the test period. The
    test dates are common to the entities: the last ``test_size`` dates of
    the panel (default 1), ``test_size`` dates from a single test date, or
    exactly the given dates when there are several (``test_size`` unused).
    Each tested entity trains on its dates up to its last test date, the last
    ``max(max_train_size, span of its test)`` of them with a window. An entity
    holding none of the test dates is warned about and not tested; it trains
    on its dates up to the last test date. Integer positions and
    ``(entity..., date)`` tuples test and train their own entities only.

    Args:
        n_splits: Unused by a panel (at least 2, as for sklearn splitters).
        test_indices: Dates (common to the entities), integer positions or
            ``(entity..., date)`` tuples; see ``InSampleSplit``.
        max_train_size: Maximum number of training dates per entity, or None.
        test_size: Number of test dates per entity.

    Examples:
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        >>> index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'v': range(8)}, index=index).drop(('B', dates[3]))
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore')
        ...     folds = [(train.tolist(), test.tolist()) for train, test in PanelInSampleSplit().split(X)]
        >>> folds
        [([0, 1, 2, 3, 4, 5, 6], [3])]
    """

    # Méthode principale de séparation des données de panel
    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate the in-sample ``(train, test)`` positions.

        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity..., date).
            y (array-like, optional): Target; a pandas target must have the index of ``X``.
            groups (array-like, optional): Entity of each row; deduced from the index when None.

        Returns:
            Generator of one ``(train_positions, test_positions)`` pair.

        Raises:
            ValueError: If ``X`` is not panel data or is empty, or for any
                error of ``InSampleSplit.split``.
        """
        # Extraction des groupes depuis la structure du panel si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)

        # Appel de la méthode parente avec les groupes pour le comportement panel
        return super().split(X, y, groups)


# Classe mixin des découpages entité par entité
class _PerEntitySplitMixin:
    """Per-entity splitting shared by the two ``PerEntity`` cross-validators.

    Default folds and dates cut each entity on its own dates, with the
    splitter of the global class (``_global_class``) applied to that entity
    alone. Positions and ``(entity..., date)`` tuples, which already open
    folds limited to one entity, reuse the global folds.
    """

    # Classe globale dont les paramètres et le découpage sont repris
    _global_class = None

    # Méthode de calcul du nombre de plis
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds that ``split`` produces on ``X``.

        The count depends on the entities and on their dates, so the data is
        required (``GridSearchCV`` passes it).

        Args:
            X: Panel data the folds would be computed on.
            y: Target, as for ``split``.
            groups: Entity of each row, as for ``split``.

        Returns:
            Number of per-entity folds.

        Raises:
            ValueError: If ``X`` is None, or if ``split`` refuses the data.
        """
        # Sans données, les entités et leurs dates sont inconnues
        if X is None:
            raise ValueError(
                f"{type(self).__name__}.get_n_splits requires X: the number of folds depends on the "
                f"entities and their dates"
            )
        # Décompte exact des plis, sans répéter les avertissements de split
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return sum(1 for _ in self.split(X, y, groups))

    # Méthode de séparation des données entité par entité
    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate one ``(train, test)`` pair per entity and per split.

        Folds come split by split (in date order), and within a split entity
        by entity, in the order of their first row in ``X``.

        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity..., date).
            y (array-like, optional): Target; a pandas target must have the index of ``X``.
            groups (array-like, optional): Entity of each row; deduced from the index when None.

        Returns:
            Generator of ``(train_positions, test_positions)`` integer arrays,
            each holding the rows of one entity.

        Raises:
            ValueError: If ``X`` is not panel data or is empty, if a
                parameter or ``groups`` is invalid, or if a test date is held
                by no entity.
        """
        # Extraction des groupes si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)
        # Contrôle des paramètres, de la cible et des groupes dès l'appel ; plis globaux paresseux
        global_folds = super().split(X, y, groups)
        parsed = _parse_test_indices(self.test_indices)
        # Positions et couples (entité, date) : plis globaux déjà limités à une entité
        if parsed is not None and parsed[0] != _DATES:
            return self._split_global_folds(global_folds, groups)
        # Plis par défaut et dates : découpage sur les dates propres à chaque entité
        return self._split_each_entity(X, groups, parsed)

    # Méthode auxiliaire de répartition des plis globaux par entité
    def _split_global_folds(self, global_folds, groups):
        """Yield each global fold restricted to each of its tested entities."""
        # Code de l'entité de chaque ligne (entités scalaires ou tuples), par
        # ordre de première apparition
        codes, uniques = pd.factorize(_labels_array(groups), sort=False)
        for train_indices, test_indices in global_folds:
            train_codes, test_codes = codes[train_indices], codes[test_indices]
            for code in range(len(uniques)):
                entity_test = test_indices[test_codes == code]
                # Entité sans ligne de test dans ce pli : pas de pli pour elle
                if len(entity_test) > 0:
                    yield train_indices[train_codes == code], entity_test

    # Méthode auxiliaire de découpage de chaque entité sur ses propres dates
    def _split_each_entity(self, X, groups, parsed):
        """Yield the folds of every entity, each cut on its own dates."""
        # Indexation
        (X,) = indexable(X)
        # Panel vide : aucune entité à découper
        if _num_samples(X) == 0:
            raise ValueError("Cannot split an empty dataset: number of samples=0")
        # Contrôle et tri des données ; lignes de chaque entité dans l'ordre chronologique
        labels = _labels_array(groups)
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, labels)
        sorted_rows = _precompute_group_mappings(groups_sorted)
        # Date demandée absente de toutes les entités : refus
        if parsed is not None:
            _resolve_test_positions(self.test_indices, X_sorted)
        # Plis rangés par numéro de découpage, entités dans l'ordre de leur première ligne
        slots: dict = {}
        for entity in _precompute_group_mappings(labels):
            rows = sort_mapping[sorted_rows[entity]]
            for slot, (train, test) in self._entity_folds(entity, X.iloc[rows], labels[rows], parsed):
                slots.setdefault(slot, []).append((rows[train], rows[test]))
        for slot in sorted(slots):
            yield from slots[slot]

    # Méthode auxiliaire de construction du splitter global d'une entité
    def _global_splitter(self, **overrides):
        """Return the global splitter with the parameters of ``self``, some overridden."""
        names = [name for name in inspect.signature(self._global_class.__init__).parameters if name != 'self']
        kwargs = {name: getattr(self, name) for name in names}
        kwargs.update(overrides)
        return self._global_class(**kwargs)

    # Méthode auxiliaire des plis d'une entité
    def _entity_folds(self, entity, X_entity, groups_entity, parsed):
        """Yield ``(slot, (train, test))`` for one entity, positions within ``X_entity``.

        ``slot`` numbers the split the fold belongs to: the default fold, or
        the requested date, it is cut for.
        """
        # Plis par défaut : les dernières dates de l'entité
        if parsed is None:
            try:
                folds = list(self._global_splitter().split(X_entity, None, groups_entity))
            except ValueError as error:
                # Entité trop courte pour porter les plis : signalée et ignorée
                warnings.warn(f"Entity {entity!r} is too short for the default folds and gets none: {error}")
                return
            yield from enumerate(folds)
            return
        # Dates demandées : seules celles que porte l'entité
        dates = _time_level(X_entity)
        items = parsed[1]
        held = []
        for slot, item in enumerate(items):
            if np.any(dates == _normalize_label(dates, item)):
                held.append(slot)
            else:
                warnings.warn(f"Cannot find test period '{item}' for entity '{entity}'; the entity gets no fold for it")
        yield from self._date_folds(entity, X_entity, groups_entity, items, held)


# Classe de validation croisée out-of-sample par entité
class PanelOutOfSampleSplitPerEntity(_PerEntitySplitMixin, PanelOutOfSampleSplit):
    """Out-of-sample cross-validator cutting each entity of a panel on its own dates.

    Each entity is split as ``PanelOutOfSampleSplit`` would split it alone,
    and every fold holds the rows of one entity:

    - without ``test_indices``, the last ``n_splits`` blocks of
      ``test_size`` dates of each entity; an entity too short for them is
      warned about and gets no fold;
    - with dates, one fold per date and per entity holding it; an entity
      lacking a date gets no fold for it (warned), as does an entity with no
      date left for training before it;
    - with positions or ``(entity..., date)`` tuples, the folds of
      ``PanelOutOfSampleSplit``, each already limited to one entity.

    Folds come split by split, and within a split entity by entity in the
    order of their first row. ``get_n_splits`` requires ``X``.

    Args:
        n_splits: Number of default folds per entity (at least 2).
        test_indices: Dates, integer positions or ``(entity..., date)`` tuples.
        max_train_size: Maximum number of training dates, or None.
        test_size: Number of test dates per fold.
        gap: Number of dates skipped between training and test.

    Examples:
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        >>> index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'v': range(8)}, index=index).drop(('B', dates[3]))
        >>> splitter = PanelOutOfSampleSplitPerEntity(n_splits=2)
        >>> [(train.tolist(), test.tolist()) for train, test in splitter.split(X)]
        [([0, 1], [2]), ([4], [5]), ([0, 1, 2], [3]), ([4, 5], [6])]
        >>> splitter.get_n_splits(X)
        4
    """

    _global_class = PanelOutOfSampleSplit

    # Méthode auxiliaire des plis d'une entité aux dates demandées
    def _date_folds(self, entity, X_entity, groups_entity, items, held):
        """Yield one fold per held date; a date leaving no training date is skipped."""
        for slot in held:
            try:
                (fold,) = self._global_splitter(test_indices=[items[slot]]).split(X_entity, None, groups_entity)
            except ValueError as error:
                # Aucune date d'apprentissage avant la date de test : signalée et ignorée
                warnings.warn(f"Entity {entity!r} gets no fold for '{items[slot]}': {error}")
                continue
            yield slot, fold


# Classe de validation croisée in-sample par entité
class PanelInSampleSplitPerEntity(_PerEntitySplitMixin, PanelInSampleSplit):
    """In-sample cross-validator testing each entity of a panel on its own dates.

    Each entity is split as ``PanelInSampleSplit`` would split it alone,
    giving one fold per entity whose training set contains its test dates:

    - without ``test_indices``, the last ``test_size`` dates of each entity;
    - with dates, the dates the entity holds (``test_size`` dates from a
      single requested date, one row per date when several are requested);
      an entity holding none of them gets no fold (warned);
    - with positions or ``(entity..., date)`` tuples, the fold of
      ``PanelInSampleSplit``, split by entity.

    Entities come in the order of their first row. ``get_n_splits`` requires
    ``X``.

    Args:
        n_splits: Unused by a panel (at least 2, as for sklearn splitters).
        test_indices: Dates, integer positions or ``(entity..., date)`` tuples.
        max_train_size: Maximum number of training dates per entity, or None.
        test_size: Number of test dates per entity.

    Examples:
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        >>> index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'v': range(8)}, index=index).drop(('B', dates[3]))
        >>> [(train.tolist(), test.tolist()) for train, test in PanelInSampleSplitPerEntity().split(X)]
        [([0, 1, 2, 3], [3]), ([4, 5, 6], [6])]
    """

    _global_class = PanelInSampleSplit

    # Méthode auxiliaire du pli d'une entité aux dates demandées
    def _date_folds(self, entity, X_entity, groups_entity, items, held):
        """Yield the single fold of the held dates, if any."""
        # Aucune date demandée portée par l'entité : pas de pli
        if not held:
            return
        overrides = {'test_indices': [items[slot] for slot in held]}
        # Plusieurs dates demandées : une ligne par date, même si l'entité n'en porte qu'une
        if len(items) > 1 and len(held) == 1:
            overrides['test_size'] = 1
        (fold,) = self._global_splitter(**overrides).split(X_entity, None, groups_entity)
        yield 0, fold
