"""Hierarchy structures of ``HierarchicalForecastAdapter``: parsing and summing matrices.

``parse_hierarchy`` reads the ``hierarchy`` parameter against the names of the
index levels and returns a ``HierarchySpec``: either a list of lists of index
levels (the index levels carry the hierarchy, aggregate rows hold an aggregate
label in their absent levels) or a ``{parent: [children]}`` mapping on the labels
of one node level. The other non-date levels identify independent copies of the
hierarchy. ``Structure`` holds the summing matrix, the level tags and the node
keys of one copy. Pure numpy and pandas: no third-party package is imported.
"""
# Importation des modules
# Modules de base
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd

# tsforecast
from tsforecast.adapters._base import _preview

# Préfixe des niveaux d'une hiérarchie donnée par un arbre (profondeur des nœuds)
DEPTH_LEVEL_PREFIX = "depth_"
# Séparateur des noms de niveaux d'index dans le nom d'un niveau de la hiérarchie
LEVEL_NAME_SEPARATOR = "/"


# Fonction auxiliaire de tri déterministe de clés
def _sorted(keys: Iterable[Hashable]) -> List[Hashable]:
    """Sort keys when they are comparable, keep their order otherwise.

    Args:
        keys: Keys to sort, in any iterable.

    Returns:
        The sorted keys, or the keys in their input order when they cannot be
        compared with each other (mixed types).
    """
    # Matérialisation des clés : un générateur ne se parcourt qu'une fois
    keys = list(keys)
    # Tri des clés : ordre déterministe, indépendant de celui des lignes
    try:
        return sorted(keys)
    except TypeError:
        # Étiquettes non comparables entre elles (types mêlés) : ordre d'entrée conservé
        return keys


# Fonction auxiliaire de contrôle d'une étiquette
def _is_missing(label: Any) -> bool:
    """Whether a scalar label is missing (None, NaN, NaT).

    Args:
        label: Index label.

    Returns:
        True for a missing scalar label; False otherwise, non-scalar labels
        (tuples, lists) included.
    """
    # Réduction du test de valeur manquante à un booléen
    try:
        return bool(pd.isna(label))
    except (TypeError, ValueError):
        # Étiquette non scalaire (tuple, liste) : considérée comme renseignée
        return False


# Classe de description d'une copie de la hiérarchie
@dataclass(frozen=True, eq=False)
class Structure:
    """Summing matrix, levels and node keys of one copy of a hierarchy.

    Nodes are ordered as hierarchicalforecast expects them: aggregate nodes
    first (most aggregated level first), bottom nodes last, so that the last
    ``n_bottom`` rows of ``summing`` form an identity matrix.

    Attributes:
        node_keys: Key of each node, in row order of ``summing``: the label
            of the node level, or the tuple of the values of the hierarchy
            levels (aggregate label in the absent levels).
        summing: Summing matrix of shape ``(n_nodes, n_bottom)`` (read-only).
        levels: Names of the levels of the hierarchy, most aggregated first.
        node_levels: Position in ``levels`` of the level of each node
            (read-only ``np.intp`` array).

    Examples:
        >>> structure = tree_structure({'a': ['b', 'c'], 'c': ['d', 'e']})
        >>> structure.node_keys, structure.levels
        (('a', 'c', 'b', 'd', 'e'), ('depth_0', 'depth_1', 'depth_2'))
        >>> structure.summing.astype(int).tolist()
        [[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """

    # Clé de chaque nœud, dans l'ordre des lignes de la matrice de sommation
    node_keys: Tuple[Hashable, ...]
    # Matrice de sommation (nœuds × feuilles)
    summing: np.ndarray
    # Noms des niveaux de la hiérarchie, du plus agrégé au plus fin
    levels: Tuple[str, ...]
    # Position, dans 'levels', du niveau de chaque nœud
    node_levels: np.ndarray

    # Nombre de nœuds (lignes de la matrice)
    @property
    def n_nodes(self) -> int:
        """Number of nodes (rows of the summing matrix)."""
        return self.summing.shape[0]

    # Nombre de feuilles (colonnes de la matrice)
    @property
    def n_bottom(self) -> int:
        """Number of bottom nodes (columns of the summing matrix)."""
        return self.summing.shape[1]

    # Identifiants des séries transmis à hierarchicalforecast
    @property
    def ids(self) -> np.ndarray:
        """Series identifiers given to hierarchicalforecast: the node positions as strings."""
        # Positions des nœuds converties en chaînes : hierarchicalforecast refuse des identifiants entiers
        return np.array([str(position) for position in range(self.n_nodes)], dtype=object)

    # Positions des nœuds de chaque niveau
    def level_positions(self) -> Dict[str, np.ndarray]:
        """Return, for each level, the positions of its nodes.

        Returns:
            Mapping level name -> ``np.intp`` array of node positions.
        """
        # Dictionnaire nom du niveau -> positions de ses nœuds
        return {
            name: np.flatnonzero(self.node_levels == position)
            for position, name in enumerate(self.levels)
        }

    # Tags au format hierarchicalforecast
    def tags(self) -> Dict[str, np.ndarray]:
        """Return the level tags in the format of hierarchicalforecast.

        Returns:
            Mapping level name -> identifiers (``ids``) of its nodes.
        """
        # Identifiants de tous les nœuds
        ids = self.ids
        # Dictionnaire nom du niveau -> identifiants de ses nœuds
        return {name: ids[positions] for name, positions in self.level_positions().items()}

    # Matrice de sommation au format hierarchicalforecast
    def summing_frame(self) -> pd.DataFrame:
        """Return the summing matrix as the ``S_df`` frame of hierarchicalforecast.

        Returns:
            Frame with a ``unique_id`` column (``ids``) and one column per
            bottom node, named by its identifier.
        """
        # Identifiants de tous les nœuds
        ids = self.ids
        # Colonnes nommées par les identifiants des feuilles (dernières lignes de la matrice)
        frame = pd.DataFrame(self.summing, columns=list(ids[-self.n_bottom:]))
        # Ajout en tête de la colonne des identifiants des lignes
        frame.insert(0, "unique_id", ids)
        return frame

    # Localisation de nœuds
    def locate(self, keys: Sequence[Hashable]) -> np.ndarray:
        """Return the position of each key among the nodes, ``-1`` when unknown.

        Args:
            keys: Node keys to locate.

        Returns:
            ``np.intp`` array of positions.

        Examples:
            >>> tree_structure({'a': ['b', 'c']}).locate(['c', 'z', 'a']).tolist()
            [2, -1, 0]
        """
        # Dictionnaire clé de nœud -> position
        lookup = {key: position for position, key in enumerate(self.node_keys)}
        # Position de chaque clé demandée, -1 pour une clé inconnue
        return np.array([lookup.get(key, -1) for key in keys], dtype=np.intp)

    # Feuilles toutes au niveau le plus fin
    def bottom_is_finest_level(self) -> bool:
        """Whether the finest level holds exactly the bottom nodes.

        Reconcilers that walk down the levels (top-down, middle-out) read the
        finest level as the bottom of the hierarchy: a tree whose leaves lie
        at different depths does not satisfy this condition.

        Returns:
            True when the nodes of the finest level are exactly the bottom nodes.
        """
        # Positions des nœuds du niveau le plus fin
        finest = np.flatnonzero(self.node_levels == len(self.levels) - 1)
        # Positions des feuilles : dernières lignes de la matrice
        bottom = np.arange(self.n_nodes - self.n_bottom, self.n_nodes)
        # Comparaison des deux ensembles de positions
        return np.array_equal(np.sort(finest), bottom)


# Fonction de construction d'une structure figée
def _make_structure(
    node_keys: Sequence[Hashable],
    summing: np.ndarray,
    levels: Sequence[str],
    node_levels: Sequence[int],
) -> Structure:
    """Build a ``Structure`` with read-only arrays.

    Args:
        node_keys: Key of each node, aggregates first, bottom nodes last.
        summing: Summing matrix, one row per node and one column per bottom node.
        levels: Names of the levels of the hierarchy, from top to bottom.
        node_levels: Position in ``levels`` of the level of each node.

    Returns:
        The frozen structure, with read-only copies of ``summing`` and
        ``node_levels``.
    """
    # Copie de la matrice en lecture seule : une structure ajustée n'est jamais modifiée
    summing = np.array(summing, dtype=np.float64)
    summing.setflags(write=False)
    # Copie en lecture seule du niveau de chaque nœud
    node_levels = np.array(node_levels, dtype=np.intp)
    node_levels.setflags(write=False)
    # Construction de la structure
    return Structure(
        node_keys=tuple(node_keys),
        summing=summing,
        levels=tuple(levels),
        node_levels=node_levels,
    )


# Fonction de construction d'une structure depuis un arbre
def tree_structure(tree: Mapping) -> Structure:
    """Build the structure of a ``{parent: [children]}`` hierarchy.

    Each child has a single parent; the roots are the parents that are no
    child, the leaves the children that are no parent. Levels are the depths
    (``'depth_0'`` for the roots); leaves may lie at different depths.

    Args:
        tree: Mapping from each aggregate node to the list of its children.

    Returns:
        The structure, aggregates ordered by depth then by order of appearance
        in ``tree``, leaves by order of appearance.

    Raises:
        ValueError: If ``tree`` is empty, if a children list is empty or holds
            duplicates, if a label is missing, if a child has several parents,
            or if the mapping has a cycle.

    Examples:
        >>> tree_structure({'total': ['x', 'y']}).node_levels.tolist()
        [0, 1, 1]
    """
    # Vérification de la forme de l'arbre
    if not isinstance(tree, Mapping) or not tree:
        raise ValueError("A {parent: [children]} hierarchy must be a non-empty mapping")
    # Dictionnaire enfant -> parent
    parent_of: Dict[Hashable, Hashable] = {}
    # Dictionnaire parent -> liste de ses enfants
    children_of: Dict[Hashable, List[Hashable]] = {}
    # Nœuds dans leur ordre de première apparition (dictionnaire employé comme ensemble ordonné)
    order: Dict[Hashable, None] = {}
    # Parcours des branches de l'arbre : contrôle puis enregistrement des liens de parenté
    for parent, children in tree.items():
        # Vérification que les enfants forment une liste non vide
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence) or not children:
            raise ValueError(f"The children of {parent!r} must be a non-empty list, got {children!r}")
        # Conversion en liste et refus des enfants répétés
        children = list(children)
        if len(set(children)) != len(children):
            raise ValueError(f"The children of {parent!r} hold duplicates: {children!r}")
        # Vérification de l'absence d'étiquette manquante
        for label in [parent, *children]:
            if _is_missing(label):
                raise ValueError("A hierarchy label is missing (None or NaN)")
        # Parcours des enfants : un seul parent admis par nœud
        for child in children:
            if child in parent_of:
                raise ValueError(
                    f"{child!r} has several parents ({parent_of[child]!r} and {parent!r}): a "
                    "{parent: [children]} hierarchy is a tree; give a grouped structure as a list "
                    "of index levels"
                )
            # Enregistrement du lien de parenté
            parent_of[child] = parent
        # Enregistrement des enfants et de l'ordre d'apparition des nœuds
        children_of[parent] = children
        order.update(dict.fromkeys([parent, *children]))

    # Identification des racines : parents qui ne sont l'enfant d'aucun nœud
    roots = [node for node in children_of if node not in parent_of]
    if not roots:
        raise ValueError("The hierarchy has a cycle: every parent is also a child")
    # Dictionnaire nœud -> profondeur, initialisé avec les racines
    depth: Dict[Hashable, int] = {root: 0 for root in roots}
    # Nœuds de la profondeur en cours de parcours
    frontier = list(roots)
    # Parcours en largeur : une profondeur par itération
    while frontier:
        # Nœuds de la profondeur suivante
        following = []
        # Parcours des enfants : profondeur du parent augmentée d'un
        for node in frontier:
            for child in children_of.get(node, []):
                depth[child] = depth[node] + 1
                following.append(child)
        # Passage à la profondeur suivante
        frontier = following
    # Nœud jamais atteint depuis une racine : membre d'un cycle
    if len(depth) != len(order):
        cycle = [node for node in order if node not in depth]
        raise ValueError(f"The hierarchy has a cycle through {_preview(cycle)}")

    # Feuilles : nœuds sans enfant, dans leur ordre d'apparition
    leaves = [node for node in order if node not in children_of]
    # Dictionnaire feuille -> colonne de la matrice de sommation
    leaf_position = {leaf: position for position, leaf in enumerate(leaves)}
    # Mémoire des feuilles descendantes : nœud -> colonnes de ses feuilles
    below: Dict[Hashable, List[int]] = {}

    # Fonction récursive de collecte des feuilles d'un nœud
    def collect(node: Hashable) -> List[int]:
        """Return the summing-matrix columns of the leaves below a node.

        Args:
            node: Node of the tree.

        Returns:
            Columns of the leaves below ``node`` (the node itself for a leaf).
        """
        # Calcul au premier passage seulement
        if node not in below:
            if node in leaf_position:
                # Cas d'une feuille : elle-même
                below[node] = [leaf_position[node]]
            else:
                # Cas d'un agrégat : réunion des feuilles de ses enfants
                below[node] = [leaf for child in children_of[node] for leaf in collect(child)]
        return below[node]

    # Agrégats triés par profondeur (tri stable : ordre d'apparition conservé)
    aggregates = sorted((node for node in order if node in children_of), key=lambda node: depth[node])
    # Ordre des lignes de la matrice : agrégats, puis feuilles
    nodes = aggregates + leaves
    # Initialisation de la matrice de sommation (nœuds × feuilles)
    summing = np.zeros((len(nodes), len(leaves)))
    # Parcours des lignes : un 1 sur chaque feuille descendant du nœud
    for row, node in enumerate(nodes):
        summing[row, collect(node)] = 1.0
    # Noms des niveaux : un par profondeur
    levels = [f"{DEPTH_LEVEL_PREFIX}{d}" for d in range(max(depth.values()) + 1)]
    # Construction de la structure
    return _make_structure(nodes, summing, levels, [depth[node] for node in nodes])


# Classe de description de la hiérarchie et des niveaux d'index
@dataclass(frozen=True, eq=False)
class HierarchySpec:
    """The ``hierarchy`` parameter read against the levels of an index.

    Attributes:
        kind: ``'levels'`` (list of lists of index levels) or ``'tree'``
            (``{parent: [children]}`` mapping on a node level).
        level_names: Names of the levels of the hierarchy, most aggregated
            first (the keys of ``model_combination``).
        node_levels: Positions of the index levels carrying the node key (one
            level for a tree).
        copy_levels: Positions of the other non-date index levels, which
            identify independent copies of the hierarchy.
        copy_names: Names of the copy levels.
        node_names: Names of the node levels.
        entries: For ``'levels'``: the levels of each level of the hierarchy,
            as positions in the node key, most aggregated first.
        aggregate_label: For ``'levels'``: label of the absent levels of an
            aggregate row.
        tree: For ``'tree'``: the structure shared by every copy.
    """

    # Forme de la hiérarchie : 'levels' ou 'tree'
    kind: str
    # Noms des niveaux de la hiérarchie, du plus agrégé au plus fin
    level_names: Tuple[str, ...]
    # Positions des niveaux d'index portant la clé de nœud
    node_levels: Tuple[int, ...]
    # Positions des niveaux d'index identifiant les copies
    copy_levels: Tuple[int, ...]
    # Noms des niveaux de copie
    copy_names: Tuple[Hashable, ...]
    # Noms des niveaux de nœud
    node_names: Tuple[Hashable, ...]
    # Forme 'levels' : pour chaque niveau de la hiérarchie, positions de ses niveaux dans la clé de nœud
    entries: Tuple[Tuple[int, ...], ...] = ()
    # Forme 'levels' : étiquette des niveaux agrégés d'une ligne
    aggregate_label: Any = None
    # Forme 'tree' : structure commune à toutes les copies
    tree: Optional[Structure] = None

    # Clés de copie et de nœud d'une combinaison de niveaux d'index
    def split_key(self, entity: tuple) -> Tuple[tuple, Hashable]:
        """Split the non-date labels of a row into its copy key and its node key.

        Args:
            entity: Labels of the non-date index levels of a row.

        Returns:
            ``(copy_key, node_key)``: tuple of the copy labels, and the node
            label (tree) or the tuple of the hierarchy labels (levels).
        """
        # Étiquettes des niveaux de copie
        copy_key = tuple(entity[position] for position in self.copy_levels)
        if self.kind == "tree":
            # Arbre : la clé de nœud est l'étiquette du niveau de nœuds
            return copy_key, entity[self.node_levels[0]]
        # Niveaux d'index : la clé de nœud réunit les étiquettes des niveaux de la hiérarchie
        return copy_key, tuple(entity[position] for position in self.node_levels)

    # Structure d'une copie depuis ses nœuds observés
    def build_structure(self, node_keys: Sequence[Hashable]) -> Structure:
        """Return the structure of a copy whose rows hold ``node_keys``.

        A tree gives its own structure; the nodes must be among its labels.
        With levels, the bottom nodes are the observed keys of the finest
        level and the aggregates are implied by projecting them on each level;
        an observed aggregate must be implied by at least one bottom node.

        Args:
            node_keys: Distinct node keys of the rows of the copy.

        Returns:
            The structure of the copy.

        Raises:
            ValueError: If a node is not in the tree, if a row matches no
                level of the hierarchy, or if an aggregate row has no bottom
                row below it.
        """
        # Cas d'un arbre : structure connue, seuls les nœuds observés sont contrôlés
        if self.kind == "tree":
            # Nœuds observés absents de l'arbre
            unknown = [key for key, position in zip(node_keys, self.tree.locate(node_keys)) if position < 0]
            if unknown:
                raise ValueError(f"Nodes absent from the hierarchy: {_preview(unknown)}")
            return self.tree
        # Cas de niveaux d'index : structure déduite des nœuds observés
        return self._levels_structure(node_keys)

    # Niveau de la hiérarchie d'une clé de nœud
    def entry_of(self, node_key: tuple) -> int:
        """Return the position in ``entries`` of the level of a node key, ``-1`` if none.

        Args:
            node_key: Labels of a node on the node levels of the index.

        Returns:
            Position of the level whose levels are exactly the labels of
            ``node_key`` that differ from the aggregate label, or ``-1``.
        """
        # Positions des niveaux renseignés (différents de l'étiquette d'agrégat)
        present = tuple(
            position for position, label in enumerate(node_key) if not _same_label(label, self.aggregate_label)
        )
        # Recherche du niveau de la hiérarchie formé exactement de ces positions
        for position, entry in enumerate(self.entries):
            if present == entry:
                return position
        # Aucun niveau correspondant
        return -1

    # Structure d'une hiérarchie donnée par des niveaux d'index
    def _levels_structure(self, node_keys: Sequence[tuple]) -> Structure:
        """Structure implied by the bottom nodes of a hierarchy given by index levels.

        Args:
            node_keys: Distinct node keys observed in the rows.

        Returns:
            The structure whose bottom nodes are the observed keys of the
            finest level and whose aggregates are their projections on each
            coarser level.

        Raises:
            ValueError: If a key matches no level, if no key is on the finest
                level, or if an observed aggregate has no bottom node below it.
        """
        # Position du niveau le plus fin
        finest = len(self.entries) - 1
        # Niveau de la hiérarchie de chaque clé observée
        entry_of_key = [self.entry_of(key) for key in node_keys]
        # Vérification que chaque clé appartient à un niveau
        unmatched = [key for key, entry in zip(node_keys, entry_of_key) if entry < 0]
        if unmatched:
            raise ValueError(
                f"Rows with labels {_preview(unmatched)} on the levels {list(self.node_names)} match no level of the "
                f"hierarchy: an aggregate row holds {self.aggregate_label!r} on the levels it aggregates"
            )
        # Feuilles : clés observées du niveau le plus fin, triées
        bottoms = _sorted(key for key, entry in zip(node_keys, entry_of_key) if entry == finest)
        # Vérification de la présence d'au moins une feuille
        if not bottoms:
            raise ValueError("The rows hold no node of the finest level of the hierarchy")
        # Dictionnaire feuille -> colonne de la matrice de sommation
        bottom_position = {key: position for position, key in enumerate(bottoms)}

        # Agrégats impliqués par la projection des feuilles sur chaque niveau : clé de chaque agrégat
        nodes: List[tuple] = []
        # Niveau de chaque agrégat
        node_levels: List[int] = []
        # Colonnes des feuilles sommées par chaque agrégat
        rows: List[List[int]] = []
        # Parcours des niveaux agrégés (tous sauf le plus fin)
        for entry_position, entry in enumerate(self.entries[:-1]):
            # Dictionnaire agrégat -> colonnes de ses feuilles, pour le niveau courant
            members: Dict[tuple, List[int]] = {}
            # Parcours des feuilles : projection de chacune sur le niveau courant
            for key in bottoms:
                # Clé de l'agrégat : étiquettes conservées sur les positions du niveau, étiquette d'agrégat ailleurs
                projected = tuple(
                    label if position in entry else self.aggregate_label for position, label in enumerate(key)
                )
                members.setdefault(projected, []).append(bottom_position[key])
            # Ajout des agrégats du niveau, dans un ordre déterministe
            for projected in _sorted(members):
                nodes.append(projected)
                node_levels.append(entry_position)
                rows.append(members[projected])
        # Vérification que chaque agrégat observé est impliqué par une feuille
        implied = set(nodes)
        orphans = [key for key, entry in zip(node_keys, entry_of_key) if entry != finest and key not in implied]
        if orphans:
            raise ValueError(f"Aggregate rows without any bottom row below them: {_preview(orphans)}")

        # Initialisation de la matrice de sommation (agrégats puis feuilles)
        summing = np.zeros((len(nodes) + len(bottoms), len(bottoms)))
        # Parcours des agrégats : un 1 sur chaque feuille qu'ils somment
        for row, members in enumerate(rows):
            summing[row, members] = 1.0
        # Bloc identité des feuilles, en dernières lignes
        summing[len(nodes):] = np.eye(len(bottoms))
        # Construction de la structure
        return _make_structure(
            nodes + bottoms, summing, self.level_names, node_levels + [finest] * len(bottoms)
        )


# Fonction auxiliaire de comparaison d'étiquettes
def _same_label(label: Any, reference: Any) -> bool:
    """Whether a label equals the aggregate label.

    Args:
        label: Index label.
        reference: Aggregate label.

    Returns:
        True when the labels compare equal; False when they differ or cannot
        be compared (composite label).
    """
    # Réduction de la comparaison à un booléen
    try:
        return bool(label == reference)
    except (TypeError, ValueError):
        # Comparaison impossible (étiquette composite) : étiquettes différentes
        return False


# Fonction auxiliaire de résolution d'un niveau d'index
def _level_position(level: Any, names: Sequence[Hashable], n_levels: int) -> int:
    """Position of a non-date index level given by name or by position.

    Args:
        level: Name of the level, or its position (negative positions count
            from the last non-date level).
        names: Names of the non-date index levels.
        n_levels: Number of non-date index levels.

    Returns:
        Non-negative position of the level among the non-date levels.

    Raises:
        ValueError: If ``level`` is neither a valid position nor the name of a
            non-date level.
    """
    # Cas d'une position : entier qui n'est pas le nom d'un niveau
    if isinstance(level, (int, np.integer)) and not isinstance(level, bool) and level not in names:
        # Position négative comptée depuis le dernier niveau précédant la date
        position = int(level) + (n_levels if level < 0 else 0)
        if not 0 <= position < n_levels:
            raise ValueError(f"node_level {level} is not a non-date level position (0 to {n_levels - 1})")
        return position
    # Cas d'un nom : vérification de son existence
    if level not in names:
        raise ValueError(f"{level!r} is not a non-date index level; levels: {list(names)}")
    # Position du niveau nommé
    return list(names).index(level)


# Fonction d'analyse du paramètre hierarchy
def parse_hierarchy(
    hierarchy: Any,
    index_names: Sequence[Hashable],
    node_level: Any = None,
    aggregate_label: Any = "Total",
) -> HierarchySpec:
    """Read the ``hierarchy`` parameter against the names of the index levels.

    Args:
        hierarchy: List of lists of index level names (the finest level, which
            holds every name, comes last once the lists are sorted by length),
            or ``{parent: [children]}`` mapping on the labels of ``node_level``.
        index_names: Names of every index level, the date level last.
        node_level: For a mapping: name or position of the level carrying the
            node labels; None for the level just before the date.
        aggregate_label: For a list of levels: label of the absent levels of an
            aggregate row.

    Returns:
        The parsed hierarchy.

    Raises:
        ValueError: If the index has no level besides the date, if
            ``hierarchy`` is neither form or is malformed, if a level name is
            unknown, or if ``node_level`` or ``aggregate_label`` is invalid for
            the form.

    Examples:
        >>> spec = parse_hierarchy([['country'], ['country', 'region']], ['market', 'country', 'region', 'date'])
        >>> spec.kind, spec.level_names, spec.copy_names, spec.node_names
        ('levels', ('country', 'country/region'), ('market',), ('country', 'region'))
        >>> parse_hierarchy({'a': ['b', 'c']}, ['node', 'date']).node_levels
        (0,)
    """
    # Noms et nombre des niveaux d'index précédant la date
    names = list(index_names[:-1])
    n_levels = len(names)
    # Vérification de la présence d'au moins un niveau portant les nœuds
    if n_levels == 0:
        raise ValueError(
            "Hierarchical data needs index levels carrying the nodes before the date level, "
            "got a single date level"
        )

    # Arbre {parent: [enfants]} sur un niveau de nœuds
    if isinstance(hierarchy, Mapping):
        # Position du niveau de nœuds : par défaut le dernier avant la date
        position = n_levels - 1 if node_level is None else _level_position(node_level, names, n_levels)
        # Construction de la structure de l'arbre
        tree = tree_structure(hierarchy)
        # Les autres niveaux identifient les copies
        copies = tuple(level for level in range(n_levels) if level != position)
        # Description d'une hiérarchie en arbre
        return HierarchySpec(
            kind="tree",
            level_names=tree.levels,
            node_levels=(position,),
            copy_levels=copies,
            copy_names=tuple(names[level] for level in copies),
            node_names=(names[position],),
            tree=tree,
        )

    # Liste de listes de niveaux d'index
    if isinstance(hierarchy, (str, bytes)) or not isinstance(hierarchy, Sequence) or not hierarchy:
        raise ValueError(
            "hierarchy must be a non-empty list of lists of index levels or a {parent: [children]} "
            f"mapping, got {hierarchy!r}"
        )
    # Refus d'un niveau de nœuds, sans objet pour cette forme
    if node_level is not None:
        raise ValueError("node_level only applies to a {parent: [children]} hierarchy")
    # Vérification de l'étiquette d'agrégat
    if _is_missing(aggregate_label) or not isinstance(aggregate_label, Hashable):
        raise ValueError(f"aggregate_label must be a hashable, non-missing label, got {aggregate_label!r}")
    # Niveaux de la hiérarchie validés, chacun en tuple de noms de niveaux d'index
    entries = []
    # Parcours des niveaux déclarés : forme, répétitions et noms inconnus
    for entry in hierarchy:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or not entry:
            raise ValueError(f"Each level of hierarchy must be a non-empty list of index levels, got {entry!r}")
        # Refus d'un niveau d'index répété
        if len(set(entry)) != len(entry):
            raise ValueError(f"The level {list(entry)} repeats an index level")
        # Refus des niveaux absents de l'index
        unknown = [level for level in entry if level not in names]
        if unknown:
            raise ValueError(f"Unknown index levels {unknown} in hierarchy; non-date levels: {names}")
        entries.append(tuple(entry))
    # Refus de deux niveaux composés des mêmes niveaux d'index
    if len({frozenset(entry) for entry in entries}) != len(entries):
        raise ValueError("hierarchy lists the same set of index levels twice")
    # Niveaux triés du plus agrégé au plus fin ; le plus fin contient tous les niveaux
    entries = sorted(entries, key=len)
    # Niveau le plus fin : le plus long
    finest = entries[-1]
    # Niveaux d'index cités mais absents du niveau le plus fin
    missing = [level for entry in entries for level in entry if level not in finest]
    if missing:
        raise ValueError(
            f"The finest level of hierarchy must hold every index level of the hierarchy, got {list(finest)}"
        )
    # Positions dans l'index des niveaux de nœud, puis des niveaux de copie
    node_levels = tuple(names.index(level) for level in finest)
    copies = tuple(level for level in range(n_levels) if level not in node_levels)
    # Description d'une hiérarchie portée par des niveaux d'index
    return HierarchySpec(
        kind="levels",
        level_names=tuple(LEVEL_NAME_SEPARATOR.join(str(level) for level in entry) for entry in entries),
        node_levels=node_levels,
        copy_levels=copies,
        copy_names=tuple(names[level] for level in copies),
        node_names=finest,
        # Niveaux exprimés en positions croissantes dans la clé de nœud, comparables aux positions
        # renseignées d'une ligne
        entries=tuple(tuple(sorted(finest.index(level) for level in entry)) for entry in entries),
        aggregate_label=aggregate_label,
    )
