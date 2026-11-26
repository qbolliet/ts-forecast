"""Temporal hierarchies of ``HierarchicalForecastAdapter``: levels, summing matrices and calendar.

``parse_temporal_hierarchy`` reads a ``{name: number of fine steps}`` mapping
(``{'year': 12, 'quarter': 3, 'month': 1}``) into a ``TemporalSpec``;
``temporal_structure`` builds the summing matrix of one period of its most
aggregated level and ``cross_structure`` the Kronecker product of a
cross-sectional and a temporal structure. ``TemporalCalendar`` places dated rows
in calendar periods and nodes; ``read_temporal_index`` and
``detect_fine_frequency`` read the temporal level of an index. Pure numpy and
pandas: no third-party package is imported.
"""
# Importation des modules
# Modules de base
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd

# tsforecast
from tsforecast.adapters._base import TemporalLayout, _preview
from tsforecast.adapters.hierarchicalforecast._structure import Structure, _level_position, _make_structure
from tsforecast.utils.parse import build_frequency_string, parse_frequency
from tsforecast.utils.time import get_period_boundaries

# Séparateur des noms de niveaux croisés (convention de get_cross_temporal_tags) : '<cs>//<te>'
CROSS_LEVEL_SEPARATOR = "//"
# Séparateur du nom d'un nœud temporel : '<niveau>-<rang dans la période>'
TEMPORAL_NODE_SEPARATOR = "-"


# Classe de description d'une hiérarchie temporelle
@dataclass(frozen=True, eq=False)
class TemporalSpec:
    """The ``temporal_hierarchy`` parameter, levels sorted most aggregated first.

    Attributes:
        levels: Names of the temporal levels, most aggregated first.
        steps: Number of fine steps covered by a node of each level (same
            order); the last one is 1.

    Examples:
        >>> hierarchy = parse_temporal_hierarchy({'month': 1, 'year': 12, 'quarter': 3})
        >>> hierarchy.levels, hierarchy.steps, hierarchy.period_steps
        (('year', 'quarter', 'month'), (12, 3, 1), 12)
        >>> hierarchy.offsets().tolist()
        [0, 1, 5]
    """

    # Noms des niveaux temporels, du plus agrégé au plus fin
    levels: Tuple[str, ...]
    # Nombre de pas fins couverts par un nœud de chaque niveau
    steps: Tuple[int, ...]

    # Nombre de pas fins d'une période du niveau le plus agrégé
    @property
    def period_steps(self) -> int:
        """Number of fine steps of a period of the most aggregated level."""
        return self.steps[0]

    # Nom du niveau le plus fin
    @property
    def finest(self) -> str:
        """Name of the finest level (one fine step per node)."""
        return self.levels[-1]

    # Nombre de nœuds de chaque niveau dans une période
    def counts(self) -> np.ndarray:
        """Number of nodes of each level in one period (``np.intp`` array)."""
        return np.array([self.period_steps // step for step in self.steps], dtype=np.intp)

    # Position du premier nœud de chaque niveau dans la structure temporelle
    def offsets(self) -> np.ndarray:
        """Position of the first node of each level in the temporal structure (``np.intp`` array)."""
        # Cumul des effectifs des niveaux précédents
        return np.concatenate([[0], np.cumsum(self.counts())[:-1]]).astype(np.intp)


# Fonction d'analyse du paramètre temporal_hierarchy
def parse_temporal_hierarchy(temporal_hierarchy: Any) -> TemporalSpec:
    """Read a ``{name: number of fine steps}`` temporal hierarchy.

    Each node of a level covers that many consecutive steps of the finest
    level (the one with 1 step); every count divides the largest one, so that
    each node lies inside one period of the most aggregated level. Levels need
    not nest into each other (``{'year': 12, 'four_months': 4, 'quarter': 3,
    'month': 1}`` gives a grouped structure).

    Args:
        temporal_hierarchy: Mapping level name -> number of fine steps.

    Returns:
        The parsed hierarchy, levels sorted most aggregated first.

    Raises:
        ValueError: If the mapping is empty, if a name is not a non-empty
            string, if a count is not a positive integer, if two levels share
            a count, if no level has 1 step, or if a count does not divide the
            largest one.

    Examples:
        >>> parse_temporal_hierarchy({'year': 4, 'semester': 2, 'quarter': 1}).levels
        ('year', 'semester', 'quarter')
    """
    # Vérification de la forme du paramètre
    if not isinstance(temporal_hierarchy, Mapping) or not temporal_hierarchy:
        raise ValueError(
            "temporal_hierarchy must be a non-empty {name: number of fine steps} mapping, "
            f"got {temporal_hierarchy!r}"
        )
    # Parcours des niveaux : contrôle du nom et du nombre de pas
    for name, step in temporal_hierarchy.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"The temporal level names must be non-empty strings, got {name!r}")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError(f"The temporal level {name!r} must cover a positive integer number of steps, got {step!r}")
    # Nombres de pas, sans répétition
    steps = [int(step) for step in temporal_hierarchy.values()]
    if len(set(steps)) != len(steps):
        raise ValueError(f"Two temporal levels cover the same number of steps: {dict(temporal_hierarchy)}")
    # Présence du niveau le plus fin
    if 1 not in steps:
        raise ValueError(f"temporal_hierarchy needs a finest level covering 1 step, got {dict(temporal_hierarchy)}")
    # Imbrication de chaque niveau dans une période du niveau le plus agrégé
    largest = max(steps)
    misfits = [name for name, step in temporal_hierarchy.items() if largest % int(step)]
    if misfits:
        raise ValueError(
            f"Each temporal level must divide the most aggregated one ({largest} steps): {misfits} do not"
        )
    # Niveaux triés du plus agrégé au plus fin
    ordered = sorted(temporal_hierarchy.items(), key=lambda item: -int(item[1]))
    return TemporalSpec(levels=tuple(name for name, _ in ordered), steps=tuple(int(step) for _, step in ordered))


# Fonction de construction de la structure d'une période
def temporal_structure(spec: TemporalSpec) -> Structure:
    """Build the summing matrix of one period of the most aggregated level.

    Node ``j`` of a level covering ``n`` steps sums the fine steps ``j * n``
    to ``(j + 1) * n - 1`` of the period; it is named ``'<level>-<j + 1>'``.

    Args:
        spec: Temporal hierarchy.

    Returns:
        The structure: levels most aggregated first, the fine steps last.

    Examples:
        >>> structure = temporal_structure(parse_temporal_hierarchy({'year': 4, 'semester': 2, 'quarter': 1}))
        >>> structure.node_keys[:3]
        ('year-1', 'semester-1', 'semester-2')
        >>> structure.summing[:3].astype(int).tolist()
        [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    """
    # Nombre de pas fins d'une période
    period = spec.period_steps
    # Clé, niveau et ligne de la matrice de chaque nœud
    keys: List[str] = []
    node_levels: List[int] = []
    rows: List[np.ndarray] = []
    # Parcours des niveaux, du plus agrégé au plus fin, puis de leurs nœuds dans la période
    for position, (name, step) in enumerate(zip(spec.levels, spec.steps)):
        for rank in range(period // step):
            # Pas fins sommés par le nœud
            row = np.zeros(period)
            row[rank * step:(rank + 1) * step] = 1.0
            keys.append(f"{name}{TEMPORAL_NODE_SEPARATOR}{rank + 1}")
            node_levels.append(position)
            rows.append(row)
    # Construction de la structure (le niveau le plus fin forme le bloc identité final)
    return _make_structure(keys, np.vstack(rows), spec.levels, node_levels)


# Classe de la structure croisée d'une copie
@dataclass(frozen=True, eq=False)
class CrossStructure:
    """Kronecker product of a cross-sectional and a temporal structure.

    Attributes:
        structure: Cross-temporal structure: node keys ``(cs_key, te_key)``,
            levels ``'<cs level>//<te level>'``.
        cross_section: Cross-sectional structure of the copy.
        positions: Position in ``structure`` of the node ``(i, j)`` at index
            ``i * n_temporal + j`` (``i``: cross-sectional node, ``j``:
            temporal node; read-only ``np.intp`` array).
    """

    # Structure croisée
    structure: Structure
    # Structure cross-section de la copie
    cross_section: Structure
    # Position du nœud croisé (i, j) à l'indice i * n_temporel + j
    positions: np.ndarray


# Fonction de construction de la structure cross-temporelle
def cross_structure(cross_section: Structure, temporal: Structure) -> CrossStructure:
    """Build the cross-temporal structure ``S_cs ⊗ S_te`` of one period.

    Row ``(i, j)`` sums the bottom cross-sectional nodes of ``i`` over the fine
    steps of ``j``. Rows are reordered so that the aggregates come first
    (grouped by cross level, cross-sectional level first) and the pairs
    (bottom node, fine step) last, in the order of the columns of the
    Kronecker product: the last rows form an identity matrix.

    Args:
        cross_section: Cross-sectional structure.
        temporal: Temporal structure.

    Returns:
        The cross structure and the position of each pair of nodes.

    Examples:
        >>> from tsforecast.adapters.hierarchicalforecast._structure import tree_structure
        >>> te = temporal_structure(parse_temporal_hierarchy({'year': 2, 'half': 1}))
        >>> cross = cross_structure(tree_structure({'t': ['x', 'y']}), te)
        >>> cross.structure.levels
        ('depth_0//year', 'depth_0//half', 'depth_1//year', 'depth_1//half')
        >>> cross.structure.node_keys[:2], cross.structure.summing.shape
        ((('t', 'year-1'), ('t', 'half-1')), (9, 4))
    """
    # Nombre de nœuds et de feuilles de chaque dimension
    n_cs, n_te = cross_section.n_nodes, temporal.n_nodes
    bottom_cs = np.arange(n_cs) >= n_cs - cross_section.n_bottom
    bottom_te = np.arange(n_te) >= n_te - temporal.n_bottom
    # Matrice de Kronecker : ligne i * n_te + j, colonne (feuille cs, pas fin)
    summing = np.kron(cross_section.summing, temporal.summing)
    # Niveau croisé de chaque couple : niveau cs majeur, niveau temporel mineur
    n_te_levels = len(temporal.levels)
    pair_levels = (cross_section.node_levels[:, None] * n_te_levels + temporal.node_levels[None, :]).ravel()
    # Couples formant le bloc identité : feuille cs et pas fin
    pair_bottom = (bottom_cs[:, None] & bottom_te[None, :]).ravel()
    # Ordre des lignes : agrégats triés (stable) par niveau croisé, puis couples du bloc identité
    aggregates = np.flatnonzero(~pair_bottom)
    order = np.concatenate([aggregates[np.argsort(pair_levels[aggregates], kind="stable")], np.flatnonzero(pair_bottom)])
    # Position de chaque couple dans l'ordre retenu
    positions = np.empty(len(order), dtype=np.intp)
    positions[order] = np.arange(len(order))
    positions.setflags(write=False)
    # Clé (nœud cs, nœud temporel) de chaque couple, dans l'ordre de Kronecker
    pair_keys = [(cs_key, te_key) for cs_key in cross_section.node_keys for te_key in temporal.node_keys]
    # Noms des niveaux croisés
    levels = [
        f"{cs_level}{CROSS_LEVEL_SEPARATOR}{te_level}"
        for cs_level in cross_section.levels
        for te_level in temporal.levels
    ]
    # Construction de la structure croisée
    structure = _make_structure(
        [pair_keys[pair] for pair in order], summing[order], levels, pair_levels[order]
    )
    return CrossStructure(structure=structure, cross_section=cross_section, positions=positions)


# Classe du placement de lignes datées dans les périodes
class TemporalPlacement(NamedTuple):
    """Period, temporal node and span of each row.

    Attributes:
        period_start: Start of the period of the most aggregated level
            holding each row.
        node: Position of the temporal node of each row in the temporal
            structure (``np.intp`` array).
        last: Date of the last fine step covered by the node of each row.
    """

    # Début de la période du niveau le plus agrégé de chaque ligne
    period_start: pd.DatetimeIndex
    # Position du nœud temporel de chaque ligne
    node: np.ndarray
    # Date du dernier pas fin couvert par le nœud de chaque ligne
    last: pd.DatetimeIndex


# Classe du calendrier d'une hiérarchie temporelle
@dataclass(frozen=True, eq=False)
class TemporalCalendar:
    """Calendar periods of a temporal hierarchy on a fine grid.

    A period of the most aggregated level is the fine grid frequency
    multiplied by its number of steps (``'12MS'`` for twelve months),
    aligned on the Unix epoch (calendar years for monthly or quarterly grids,
    days for hourly grids) or on ``origin``; it must hold exactly that many
    dates of the fine grid. A node is dated by its first fine step on a grid
    labelling its periods by their start (``'MS'``, ``'QS'``, ``'D'``,
    ``'h'``...), by its last one on a grid labelling them by their end
    (``'ME'``, ``'QE'``, ``'W-SUN'``).

    Attributes:
        spec: Temporal hierarchy.
        freq: Pandas alias of the fine grid.
        period_freq: Pandas alias of the periods of the most aggregated level.
        end_labelled: Whether the fine grid labels its periods by their end.
        origin: Start of one period of the most aggregated level, or None for
            the Unix epoch.

    Examples:
        >>> import pandas as pd
        >>> hierarchy = parse_temporal_hierarchy({'year': 4, 'quarter': 1})
        >>> calendar = TemporalCalendar.build(hierarchy, 'QE-DEC')
        >>> calendar.period_freq, calendar.end_labelled
        ('4QE-DEC', True)
        >>> placement = calendar.locate(np.array([0, 1]), pd.DatetimeIndex(['2024-12-31', '2024-06-30']))
        >>> placement.period_start.strftime('%Y-%m-%d').tolist(), placement.node.tolist()
        (['2024-01-01', '2024-01-01'], [0, 2])
    """

    # Hiérarchie temporelle
    spec: TemporalSpec
    # Alias pandas de la grille fine
    freq: str
    # Alias pandas des périodes du niveau le plus agrégé
    period_freq: str
    # Grille étiquetant ses périodes par leur fin
    end_labelled: bool
    # Début d'une période du niveau le plus agrégé, None pour l'époque Unix
    origin: Optional[pd.Timestamp]

    # Construction depuis la grille fine
    @classmethod
    def build(cls, spec: TemporalSpec, freq: str, origin: Any = None) -> "TemporalCalendar":
        """Build the calendar of ``spec`` on the fine grid ``freq``.

        Args:
            spec: Temporal hierarchy.
            freq: Pandas alias of the fine grid.
            origin: Date at which a period of the most aggregated level
                starts, or None for the Unix epoch.

        Returns:
            The calendar.

        Raises:
            ValueError: If ``origin`` is not a date.
        """
        # Décomposition de la fréquence fine
        parsed = parse_frequency(freq)
        # Période du niveau le plus agrégé : fréquence fine multipliée
        period_freq = build_frequency_string(
            parsed.freq, parsed.position, parsed.suffix, multiplier=parsed.multiplier * spec.period_steps
        )
        # Conversion de l'origine en date
        if origin is not None:
            try:
                origin = pd.Timestamp(origin)
            except (TypeError, ValueError) as error:
                raise ValueError(f"temporal_origin must be a date, got {origin!r}") from error
            if pd.isna(origin):
                raise ValueError("temporal_origin must be a date, got a missing value")
        return cls(
            spec=spec,
            freq=freq,
            period_freq=period_freq,
            # Étiquette de fin : position 'E' et semaines ancrées sur leur dernier jour
            end_labelled=parsed.position == "E" or parsed.freq == "W",
            origin=origin,
        )

    # Grilles fines des périodes contenant des dates
    def _place(self, dates: pd.DatetimeIndex) -> Tuple[List[pd.Timestamp], np.ndarray, Dict[pd.Timestamp, pd.DatetimeIndex]]:
        """Return the period start and the fine position of each date, and the grid of each period."""
        # Dictionnaire début de période -> dates de la grille fine de la période
        grids: Dict[pd.Timestamp, pd.DatetimeIndex] = {}
        # Début de période et position fine de chaque date
        starts: List[Any] = [None] * len(dates)
        positions = np.empty(len(dates), dtype=np.intp)
        # Période en cours de parcours : début, fin exclue, grille
        current: Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]] = None
        # Parcours des dates dans l'ordre chronologique : une période calculée une fois
        for position in np.argsort(dates.asi8, kind="stable"):
            date = dates[position]
            # Nouvelle période dès que la date sort de la période courante
            if current is None or not current[0] <= date < current[1]:
                start, end = get_period_boundaries(date, self.period_freq, origin=self.origin)
                grid = pd.date_range(start, end, freq=self.freq, inclusive="left")
                # Vérification du pavage de la période par la grille fine
                if len(grid) != self.spec.period_steps:
                    raise ValueError(
                        f"The '{self.freq}' grid holds {len(grid)} dates in the period {start} - {end} of the "
                        f"most aggregated temporal level, not {self.spec.period_steps}: this grid cannot carry "
                        "the temporal hierarchy"
                    )
                current = (start, end, grid)
                grids[start] = grid
            # Position de la date dans la grille de sa période
            located = current[2].get_indexer([date])[0]
            if located < 0:
                raise ValueError(
                    f"The date {date} is not on the '{self.freq}' grid of its period {current[0]} - {current[1]}"
                )
            starts[position] = current[0]
            positions[position] = located
        return starts, positions, grids

    # Placement de lignes datées
    def locate(self, level_codes: np.ndarray, dates: pd.DatetimeIndex) -> TemporalPlacement:
        """Place each row in its period and its temporal node.

        Args:
            level_codes: Position in ``spec.levels`` of the temporal level of
                each row.
            dates: Date of each row.

        Returns:
            The period, node and last fine date of each row.

        Raises:
            ValueError: If a date is not on the fine grid of its period, or if
                an aggregate row is not dated by the first (start-labelled
                grid) or the last (end-labelled grid) fine step of a node.
        """
        # Code de la date de chaque ligne et dates distinctes
        date_codes, distinct = pd.factorize(pd.DatetimeIndex(dates))
        distinct = pd.DatetimeIndex(distinct)
        # Début de période, position fine et grilles des dates distinctes
        starts, fine_positions, grids = self._place(distinct)
        # Diffusion aux lignes
        row_position = fine_positions[date_codes]
        # Nombre de pas fins du nœud de chaque ligne
        steps = np.asarray(self.spec.steps, dtype=np.intp)[level_codes]
        # Rang du nœud : un nœud est daté par son premier ou par son dernier pas fin
        anchor = row_position + 1 if self.end_labelled else row_position
        aligned = anchor % steps == 0
        rank = anchor // steps - (1 if self.end_labelled else 0)
        # Refus d'une ligne d'agrégat datée hors des dates de ses nœuds
        if not aligned.all():
            first = int(np.flatnonzero(~aligned)[0])
            side = "last" if self.end_labelled else "first"
            raise ValueError(
                f"A row of the temporal level {self.spec.levels[level_codes[first]]!r} is dated {dates[first]}, "
                f"which is not the {side} '{self.freq}' step of one of its periods: on this grid, an aggregate "
                f"is dated by the {side} fine step it covers"
            )
        # Position du nœud dans la structure temporelle
        node = self.spec.offsets()[level_codes] + rank
        # Début de période de chaque ligne
        period_start = pd.DatetimeIndex([starts[code] for code in date_codes])
        # Dernier pas fin couvert par chaque nœud
        last_position = (rank + 1) * steps - 1
        last = pd.DatetimeIndex(
            [grids[starts[code]][offset] for code, offset in zip(date_codes, last_position)]
        )
        return TemporalPlacement(period_start=period_start, node=node.astype(np.intp), last=last)


# Classe de lecture du niveau temporel d'un index
class TemporalIndex(NamedTuple):
    """Temporal level of an index read against a temporal hierarchy.

    Attributes:
        position: Position of the temporal level among the non-date levels.
        entity_levels: Temporal level code of each distinct entity of the
            layout (``np.intp`` array).
        row_levels: Temporal level code of each row (``np.intp`` array).
    """

    # Position du niveau temporel parmi les niveaux non date
    position: int
    # Code du niveau temporel de chaque entité distincte
    entity_levels: np.ndarray
    # Code du niveau temporel de chaque ligne
    row_levels: np.ndarray


# Fonction de lecture du niveau temporel d'un index
def read_temporal_index(layout: TemporalLayout, spec: TemporalSpec, temporal_level: Any = None) -> TemporalIndex:
    """Read the temporal level of each row.

    Args:
        layout: Layout of the index; its entity levels hold the temporal level.
        spec: Temporal hierarchy.
        temporal_level: Name or position of the temporal level among the
            non-date levels; None for the level just before the date.

    Returns:
        The position of the temporal level and the level code of each entity
        and row.

    Raises:
        ValueError: If the index has no level besides the date, if
            ``temporal_level`` is unknown, or if a label is not a level of
            ``spec``.
    """
    # Noms des niveaux précédant la date
    names = list(layout.entity_names)
    if not names:
        raise ValueError(
            "Temporal reconciliation needs an index level holding the temporal level of each row "
            "before the date level"
        )
    # Position du niveau temporel : par défaut le dernier avant la date
    position = len(names) - 1 if temporal_level is None else _level_position(temporal_level, names, len(names))
    # Dictionnaire nom de niveau temporel -> code
    lookup = {name: code for code, name in enumerate(spec.levels)}
    # Code du niveau temporel de chaque entité distincte (-1 pour une étiquette inconnue)
    entity_levels = np.array([lookup.get(entity[position], -1) for entity in layout.entities], dtype=np.intp)
    # Refus des étiquettes qui ne sont pas des niveaux temporels
    if (entity_levels < 0).any():
        unknown = list(dict.fromkeys(layout.entities[code][position] for code in np.flatnonzero(entity_levels < 0)))
        raise ValueError(
            f"Labels {_preview(unknown)} of the temporal index level {names[position]!r} are not levels of "
            f"temporal_hierarchy {list(spec.levels)}"
        )
    return TemporalIndex(position=position, entity_levels=entity_levels, row_levels=entity_levels[layout.codes])


# Fonction de détection de la grille fine
def detect_fine_frequency(layout: TemporalLayout, row_levels: np.ndarray, spec: TemporalSpec) -> str:
    """Detect the grid of the rows of the finest temporal level.

    Args:
        layout: Layout of the index.
        row_levels: Temporal level code of each row.
        spec: Temporal hierarchy.

    Returns:
        Pandas alias of the fine grid.

    Raises:
        ValueError: If no series of the finest level has a detectable regular
            grid, or if the series are on different grids.
    """
    # Lignes du niveau le plus fin
    fine = row_levels == len(spec.levels) - 1
    # Index réduit à ces lignes : code d'entité (une série par entité) et date
    index = pd.MultiIndex.from_arrays([layout.codes[fine], layout.dates[fine]])
    # Détection de la fréquence commune aux séries
    freq = TemporalLayout.from_index(index, detect_freq=True).freq
    if freq is None:
        raise ValueError(
            f"The grid of the finest temporal level {spec.finest!r} cannot be detected: the rows of that level "
            "need a regular grid with at least two dates in one series"
        )
    return freq

