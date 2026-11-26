"""Sklearn adapter for hierarchicalforecast reconciliation: ``HierarchicalForecastAdapter``.

``X`` holds the base forecasts of every node of a hierarchy (one column per base
model) and ``y`` the realised values, in a long layout: one row per node and date,
the node carried by index levels placed before the date. ``fit`` builds the summing
matrix of each copy of the hierarchy and keeps the training rows that reconcilers
learning from in-sample errors need; ``predict`` reconciles the base forecasts of
the rows of ``X`` date by date (cross-sectional hierarchy), or calendar period by
calendar period (temporal hierarchy, alone or crossed with a cross-sectional one),
and returns them in the row order of ``X``; ``reconcile`` and ``diagnose`` expose
every base model and reconciler.
"""
# Importation des modules
# Modules de base
import copy
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# tsforecast
from tsforecast.adapters._base import (
    TemporalLayout,
    _preview,
    check_adapter_input,
    import_optional,
    restore_row_order,
)
from tsforecast.adapters.hierarchicalforecast._structure import HierarchySpec, Structure, parse_hierarchy
from tsforecast.adapters.hierarchicalforecast._temporal import (
    CROSS_LEVEL_SEPARATOR,
    TemporalCalendar,
    TemporalSpec,
    cross_structure,
    detect_fine_frequency,
    parse_temporal_hierarchy,
    read_temporal_index,
    temporal_structure,
)

# Extra installant hierarchicalforecast
_EXTRA = "adapters-hierarchical"
# Nom de la prévision de base combinée dans les sorties de reconcile et diagnose
COMBINED_NAME = "combined"
# Combinaison des modèles de base par moyenne
_MEAN_COMBINATION = "mean"
# Colonnes des cadres de hierarchicalforecast
_ID_COL = "unique_id"
_TIME_COL = "ds"
_TARGET_COL = "y"
# Noms internes des modèles : aucun nom de colonne de l'utilisateur ne part vers hierarchicalforecast,
# qui écarte en silence un modèle dont le nom contient '-lo', '-hi' ou '-median'
_MODEL_PREFIX = "model_"
_COMBINED_INTERNAL = "model_combined"
# Colonnes d'intervalle de prévision d'un modèle de base : '<modèle>-lo-<niveau>', '<modèle>-hi-<niveau>'
_INTERVAL_COLUMN = re.compile(r"^(?P<model>.+)-(?P<side>lo|hi)-(?P<level>\d+(?:\.\d+)?)$")
# Bornes inférieure et supérieure d'un intervalle de prévision
_INTERVAL_SIDES = ("lo", "hi")
# Méthodes d'intervalle qui lisent les erreurs insample
_INSAMPLE_INTERVAL_METHODS = frozenset({"bootstrap", "permbu", "conformal"})
# Modes de réconciliation : cross-section (colonnes = dates), temporel et cross-temporel (colonnes =
# périodes du niveau temporel le plus agrégé)
_CROSS_SECTION = "cross_section"
_TEMPORAL = "temporal"
_CROSS_TEMPORAL = "cross_temporal"


# Fonction d'import différé de hierarchicalforecast
def _import_hierarchicalforecast() -> Tuple[ModuleType, ModuleType, ModuleType]:
    """Import the core, methods and utils modules of hierarchicalforecast.

    Returns:
        The modules ``hierarchicalforecast.core``, ``hierarchicalforecast.methods``
        and ``hierarchicalforecast.utils``.

    Raises:
        ImportError: If hierarchicalforecast is not installed (the message
            names the extra to install).
    """
    # Import différé : le paquet n'est exigé qu'à l'ajustement et à la prédiction
    return (
        import_optional("hierarchicalforecast.core", _EXTRA),
        import_optional("hierarchicalforecast.methods", _EXTRA),
        import_optional("hierarchicalforecast.utils", _EXTRA),
    )


# Classe des données ajustées d'une copie de la hiérarchie
@dataclass(frozen=True, eq=False)
class _CopyFit:
    """Structure and training rows of one copy of the hierarchy.

    Attributes:
        structure: Summing matrix, levels and nodes reconciled for the copy
            (cross-sectional, temporal or cross-temporal).
        cross_section: Cross-sectional structure of the copy, None in the
            temporal mode.
        positions: Cross-temporal mode: position in ``structure`` of the pair
            (cross-sectional node ``i``, temporal node ``j``) at index
            ``i * n_temporal + j``; None otherwise.
        insample: Training rows in the format of hierarchicalforecast
            (``unique_id``, ``ds``, ``y`` and one column per base model);
            None when no reconciliation may read them.
        missing_nodes: Nodes without any training row with a target.
        balanced: Whether every node has a row at every training date.
    """

    # Matrice de sommation, niveaux et nœuds réconciliés pour la copie
    structure: Structure
    # Structure cross-section de la copie (None en mode temporel)
    cross_section: Optional[Structure] = None
    # Mode cross-temporel : position du couple (nœud cs i, nœud temporel j) à l'indice i * n_temporel + j
    positions: Optional[np.ndarray] = None
    # Lignes d'entraînement au format de hierarchicalforecast
    insample: Optional[pd.DataFrame] = None
    # Nœuds sans aucune ligne d'entraînement à cible renseignée
    missing_nodes: Tuple[Hashable, ...] = ()
    # Présence d'une ligne pour chaque nœud à chaque date d'entraînement
    balanced: bool = False


# Classe de description du mode de réconciliation, fixée au fit
@dataclass(frozen=True, eq=False)
class _Design:
    """Mode, hierarchies and calendar of a fitted adapter.

    Attributes:
        mode: ``'cross_section'``, ``'temporal'`` or ``'cross_temporal'``.
        spec: Cross-sectional hierarchy read on the index levels other than
            the temporal one; None in the temporal mode.
        temporal: Temporal hierarchy; None in the cross-sectional mode.
        temporal_level: ``temporal_level`` parameter (name or position).
        temporal_name: Name of the temporal index level.
        calendar: Calendar of the temporal hierarchy on the fine grid.
        temporal_structure: Structure of one period of the most aggregated
            temporal level.
        level_names: Names of the levels of the reconciled structures.
        copy_names: Names of the copy levels (substitute name for an
            anonymous level).
    """

    # Mode de réconciliation
    mode: str
    # Hiérarchie cross-section (None en mode temporel)
    spec: Optional[HierarchySpec]
    # Hiérarchie temporelle (None en mode cross-section)
    temporal: Optional[TemporalSpec] = None
    # Paramètre temporal_level
    temporal_level: Any = None
    # Nom du niveau d'index temporel
    temporal_name: Hashable = None
    # Calendrier de la hiérarchie temporelle
    calendar: Optional[TemporalCalendar] = None
    # Structure d'une période du niveau temporel le plus agrégé
    temporal_structure: Optional[Structure] = None
    # Noms des niveaux des structures réconciliées
    level_names: Tuple[str, ...] = ()
    # Noms des niveaux de copie
    copy_names: Tuple[Hashable, ...] = ()

    # Données structurelles d'une copie depuis sa structure cross-section
    def record(self, cross_section: Optional[Structure]) -> _CopyFit:
        """Return the structures of a copy whose cross-sectional structure is ``cross_section``.

        Args:
            cross_section: Cross-sectional structure of the copy, None in the
                temporal mode.

        Returns:
            The structures of the copy in the mode of the design: the
            cross-sectional structure, the structure of one period, or their
            cross-temporal product.
        """
        # Mode cross-section : la structure cross-section est réconciliée telle quelle
        if self.mode == _CROSS_SECTION:
            return _CopyFit(structure=cross_section, cross_section=cross_section)
        # Mode temporel : structure d'une période, commune à toutes les copies
        if self.mode == _TEMPORAL:
            return _CopyFit(structure=self.temporal_structure)
        # Mode cross-temporel : produit de Kronecker des deux structures
        cross = cross_structure(cross_section, self.temporal_structure)
        return _CopyFit(structure=cross.structure, cross_section=cross_section, positions=cross.positions)


# Classe de séparation des combinaisons de niveaux d'index
@dataclass(frozen=True, eq=False)
class _EntitySplit:
    """Copy, node and temporal level of each distinct combination of index levels.

    Attributes:
        copy_keys: Distinct copy keys, in order of first appearance.
        entity_copy: Position in ``copy_keys`` of the copy of each entity.
        node_keys: Cross-sectional node key of each entity (None in the
            temporal mode).
        row_temporal: Temporal level code of each row; None in the
            cross-sectional mode.
    """

    # Clés des copies distinctes, dans l'ordre de leurs codes
    copy_keys: Tuple[tuple, ...]
    # Code de la copie de chaque combinaison distincte de niveaux d'index
    entity_copy: np.ndarray
    # Clé de nœud cross-section de chaque combinaison distincte
    node_keys: List[Hashable]
    # Code du niveau temporel de chaque ligne (None en mode cross-section)
    row_temporal: Optional[np.ndarray]


# Classe de localisation des lignes dans les copies de la hiérarchie
@dataclass(frozen=True, eq=False)
class _RowNodes:
    """Copy, node and reconciliation column of each row of a dataset.

    Attributes:
        copy_keys: Distinct copy keys, in order of first appearance.
        structures: Structure of each copy (same order).
        row_copy: Position in ``copy_keys`` of the copy of each row.
        row_node: Position of the node of each row in the structure of its copy.
        row_level: Position of the level of the node of each row.
        row_column: Column of each row: its date (cross-section) or the
            start of its period of the most aggregated temporal level.
    """

    # Clés des copies distinctes, dans l'ordre de leurs codes
    copy_keys: Tuple[tuple, ...]
    # Structure de chaque copie, dans le même ordre
    structures: Tuple[Structure, ...]
    # Code de la copie de chaque ligne
    row_copy: np.ndarray
    # Position du nœud de chaque ligne dans la structure de sa copie
    row_node: np.ndarray
    # Position du niveau du nœud de chaque ligne
    row_level: np.ndarray
    # Colonne de réconciliation de chaque ligne : date ou début de période
    row_column: pd.DatetimeIndex


# Fonction de séparation des combinaisons de niveaux d'index en copies, nœuds et niveaux temporels
def _split_entities(layout: TemporalLayout, design: _Design) -> _EntitySplit:
    """Split each distinct combination of index levels into its copy, its node and its temporal level.

    Args:
        layout: Temporal and entity structure of the rows.
        design: Hierarchies read against the index levels.

    Returns:
        The copy keys, the copy code and the node key of each distinct
        combination of index levels, and the temporal level of each row in
        the temporal modes.
    """
    # Niveau temporel de chaque combinaison et de chaque ligne (modes temporels)
    temporal = (
        read_temporal_index(layout, design.temporal, design.temporal_level) if design.temporal is not None else None
    )
    # Dictionnaire clé de copie -> code (ordre de première apparition)
    lookup: Dict[tuple, int] = {}
    # Code de copie et clé de nœud de chaque combinaison distincte de niveaux d'index
    entity_copy, node_keys = [], []
    # Une seule analyse par combinaison distincte, diffusée ensuite par les codes des lignes
    for entity in layout.entities:
        # Retrait de l'étiquette du niveau temporel
        if temporal is not None:
            entity = entity[: temporal.position] + entity[temporal.position + 1:]
        # Séparation des étiquettes de copie et de nœud (mode temporel : tous les niveaux forment la copie)
        copy_key, node_key = (entity, None) if design.spec is None else design.spec.split_key(entity)
        # Attribution d'un nouveau code à une copie jamais rencontrée
        entity_copy.append(lookup.setdefault(copy_key, len(lookup)))
        node_keys.append(node_key)
    return _EntitySplit(
        copy_keys=tuple(lookup),
        entity_copy=np.array(entity_copy, dtype=np.intp),
        node_keys=node_keys,
        row_temporal=None if temporal is None else temporal.row_levels,
    )


# Fonction de placement des lignes dans les structures de leurs copies
def _place_rows(
    layout: TemporalLayout, design: _Design, split: _EntitySplit, records: Sequence[_CopyFit]
) -> _RowNodes:
    """Locate the node and the reconciliation column of each row.

    Args:
        layout: Temporal and entity structure of the rows.
        design: Hierarchies read against the index levels.
        split: Copy and node of each distinct combination of index levels.
        records: Structures of each copy, in the order of ``split.copy_keys``.

    Returns:
        The copy, the node, the level of the node and the reconciliation
        column (date, or start of period in the temporal modes) of each row.

    Raises:
        ValueError: If a cross-sectional node is unknown to the structure of
            its copy, or if a temporal row is off its calendar.
    """
    # Placement des lignes dans les périodes et les nœuds temporels (modes temporels)
    placement = (
        design.calendar.locate(split.row_temporal, layout.dates) if split.row_temporal is not None else None
    )
    # Position du nœud cross-section de chaque combinaison de niveaux d'index
    entity_cs = np.zeros(len(split.node_keys), dtype=np.intp)
    # Parcours des copies : localisation de leurs nœuds cross-section
    for code, (copy_key, record) in enumerate(zip(split.copy_keys, records)):
        # Mode temporel : aucun nœud cross-section
        if record.cross_section is None:
            continue
        # Combinaisons de niveaux d'index appartenant à la copie et position de leurs nœuds
        members = np.flatnonzero(split.entity_copy == code)
        positions = record.cross_section.locate([split.node_keys[member] for member in members])
        # Refus des nœuds absents de la structure
        if (positions < 0).any():
            unknown = list(dict.fromkeys(split.node_keys[member] for member in members[positions < 0]))
            where = f" of the copy {copy_key!r}" if copy_key else ""
            raise ValueError(f"Nodes{where} unknown at fit: {_preview(unknown)}")
        entity_cs[members] = positions

    # Copie de chaque ligne, par le code de sa combinaison d'étiquettes
    row_copy = split.entity_copy[layout.codes]
    # Nœud de chaque ligne selon le mode
    if design.mode == _CROSS_SECTION:
        # Cross-section : le nœud de la combinaison d'étiquettes
        row_node = entity_cs[layout.codes]
    elif design.mode == _TEMPORAL:
        # Temporel : le nœud temporel de la ligne
        row_node = placement.node
    else:
        # Cross-temporel : couple (nœud cs, nœud temporel) placé dans la structure croisée de la copie
        row_node = np.empty(len(layout.codes), dtype=np.intp)
        # Nombre de nœuds temporels
        n_temporal = design.temporal_structure.n_nodes
        # Parcours des copies : position des couples de leurs lignes
        for code, record in enumerate(records):
            rows = np.flatnonzero(row_copy == code)
            row_node[rows] = record.positions[entity_cs[layout.codes[rows]] * n_temporal + placement.node[rows]]
    # Niveau du nœud de chaque ligne
    row_level = np.empty(len(row_node), dtype=np.intp)
    # Parcours des copies : niveau des nœuds de leurs lignes dans leur structure
    for code, record in enumerate(records):
        rows = row_copy == code
        row_level[rows] = record.structure.node_levels[row_node[rows]]
    return _RowNodes(
        copy_keys=split.copy_keys,
        structures=tuple(record.structure for record in records),
        row_copy=row_copy,
        row_node=row_node,
        row_level=row_level,
        # Colonne : date en cross-section, début de période dans les modes temporels
        row_column=layout.dates if placement is None else placement.period_start,
    )


# Fonction de nommage des niveaux de copie
def _copy_names(names: Sequence[Hashable], positions: Sequence[int]) -> Tuple[Hashable, ...]:
    """Names of the copy levels, ``'level_<position>'`` for an anonymous level.

    Args:
        names: Name of each copy level, None for an anonymous level.
        positions: Position of each copy level in the index.

    Returns:
        The names of the copy levels, anonymous levels named after their position.
    """
    # Nom de substitution pour un niveau sans nom
    return tuple(f"level_{position}" if name is None else name for name, position in zip(names, positions))


# Fonction de résolution des colonnes de prévisions de base et d'intervalle
def _resolve_columns(X: pd.DataFrame, forecast_columns: Any) -> Tuple[List[Hashable], Dict[tuple, Hashable]]:
    """Return the base forecast columns and the interval columns ``(model, side, level) -> column``.

    Args:
        X: Base forecasts, one column per base model, with optional interval
            columns ``'<model>-lo-<level>'`` and ``'<model>-hi-<level>'``.
        forecast_columns: Base forecast columns chosen by the user, or None
            for every column that is not an interval column.

    Returns:
        Pair ``(base, intervals)``: the base forecast columns, and the
        dictionary ``(model, side, level)`` -> interval column restricted to
        these base models.

    Raises:
        ValueError: If ``forecast_columns`` is not a list of distinct columns
            of ``X``, or if no base forecast column is left or one is not numeric.
    """
    # Colonnes de X
    columns = list(X.columns)
    # Dictionnaire colonne d'intervalle -> (modèle, borne, niveau de confiance)
    interval_of: Dict[Hashable, tuple] = {}
    # Parcours des colonnes : repérage de celles qui encadrent les prévisions d'une autre colonne
    for column in columns:
        # Lecture du suffixe '-lo-<niveau>' ou '-hi-<niveau>' (noms textuels seulement)
        match = _INTERVAL_COLUMN.match(column) if isinstance(column, str) else None
        # Colonne d'intervalle seulement si le modèle qu'elle encadre est présent
        if match is not None and match["model"] in columns:
            interval_of[column] = (match["model"], match["side"], match["level"])

    if forecast_columns is None:
        # Par défaut : toutes les colonnes hors colonnes d'intervalle
        base = [column for column in columns if column not in interval_of]
    else:
        # Colonnes choisies : liste de colonnes de X, sans répétition
        if isinstance(forecast_columns, (str, bytes)) or not isinstance(forecast_columns, Sequence):
            raise ValueError(f"forecast_columns must be a list of columns of X, got {forecast_columns!r}")
        base = list(forecast_columns)
        unknown = [column for column in base if column not in columns]
        if unknown:
            raise ValueError(f"forecast_columns {_preview(unknown)} are not columns of X")
        if len(set(base)) != len(base):
            raise ValueError(f"forecast_columns holds duplicates: {base!r}")
    # Vérification de la présence d'au moins une prévision de base
    if not base:
        raise ValueError("X holds no base forecast column")
    # Vérification du caractère numérique des prévisions de base
    non_numeric = [column for column in base if not pd.api.types.is_numeric_dtype(X[column])]
    if non_numeric:
        raise ValueError(f"The base forecast columns {_preview(non_numeric)} are not numeric")
    # Dictionnaire (modèle, borne, niveau) -> colonne, restreint aux prévisions de base
    intervals = {key: column for column, key in interval_of.items() if key[0] in base}
    return base, intervals


# Fonction de résolution de la combinaison des modèles de base
def _resolve_combination(
    model_combination: Any, level_names: Sequence[str], base: Sequence[Hashable]
) -> Optional[Any]:
    """Return None, ``'mean'`` or the base column position of each level of the hierarchy.

    Args:
        model_combination: The ``model_combination`` parameter: None,
            ``'mean'`` or a mapping level -> base forecast column.
        level_names: Names of the levels of the hierarchy.
        base: Base forecast columns.

    Returns:
        None without combination, ``'mean'`` for the mean of the base
        forecasts, or the ``np.intp`` array of the position in ``base`` of the
        column chosen for each level, in the order of ``level_names``.

    Raises:
        ValueError: If several base columns are given without combination, if
            a base column is named like the combined forecast, or if the
            combination is not a valid name or mapping of the levels.
    """
    # Cas sans combinaison : une seule prévision de base admise
    if model_combination is None:
        if len(base) != 1:
            raise ValueError(
                f"X has {len(base)} base forecast columns {_preview(base)}: combine them with "
                "model_combination ('mean' or {level: column}) or select one with forecast_columns"
            )
        return None
    # Vérification que le nom de la prévision combinée reste disponible
    if COMBINED_NAME in base:
        raise ValueError(
            f"A base forecast column is named {COMBINED_NAME!r}, the name of the combined forecast: rename it"
        )
    # Cas d'une combinaison désignée par son nom : la moyenne
    if isinstance(model_combination, str):
        if model_combination != _MEAN_COMBINATION:
            raise ValueError(
                f"model_combination must be None, 'mean' or a {{level: column}} mapping, got {model_combination!r}"
            )
        return _MEAN_COMBINATION
    # Vérification que la dernière forme admise est un dictionnaire niveau -> colonne
    if not isinstance(model_combination, Mapping):
        raise ValueError(
            f"model_combination must be None, 'mean' or a {{level: column}} mapping, got {model_combination!r}"
        )
    # Vérification que chaque niveau de la hiérarchie a une colonne, sans niveau inconnu
    missing = [level for level in level_names if level not in model_combination]
    extra = [level for level in model_combination if level not in level_names]
    if missing or extra:
        raise ValueError(
            f"model_combination must map each level of the hierarchy {list(level_names)} to a column; "
            f"missing levels: {missing}, unknown levels: {extra}"
        )
    # Vérification que les colonnes citées sont des prévisions de base
    unknown = [column for column in model_combination.values() if column not in base]
    if unknown:
        raise ValueError(f"model_combination names columns that are no base forecast: {_preview(unknown)}")
    # Position de la colonne retenue pour chaque niveau, dans l'ordre des niveaux
    positions = [list(base).index(model_combination[level]) for level in level_names]
    return np.array(positions, dtype=np.intp)


# Wrapper de la réconciliation hiérarchique dans un régresseur sklearn
class HierarchicalForecastAdapter(RegressorMixin, BaseEstimator):
    """Sklearn regressor reconciling the base forecasts of a hierarchy with hierarchicalforecast.

    Data are long: one row per node of the hierarchy and per date, the date in
    the last index level and the node in the index levels before it. ``y``
    holds the realised values of every node (bottom and aggregate nodes) and
    ``X`` the base forecasts of these rows, one column per base model,
    produced upstream. ``predict`` returns the reconciled forecasts of the rows
    of ``X``, in their order: on each date, the forecasts of the nodes become
    coherent (each aggregate equals the sum of its bottom nodes). Each date is
    reconciled on its own, so the prediction of a row does not depend on the
    gap between training and test.

    The hierarchy takes one of two forms:

    - a list of lists of index levels, as the ``spec`` of hierarchicalforecast
      (``[['country'], ['country', 'region']]``): the index levels carry the
      hierarchy; an aggregate row holds ``aggregate_label`` on the levels it
      aggregates (``('FR', 'Total', date)``); grouped structures are accepted
      (``[['country'], ['country', 'purpose'], ['country', 'region'],
      ['country', 'region', 'purpose']]``). The nodes are those of the
      training rows;
    - a ``{parent: [children]}`` mapping on the labels of the index level
      ``node_level`` (``{'a': ['b', 'c'], 'c': ['d', 'e']}``); its levels are
      the depths ``'depth_0'``, ``'depth_1'``...

    The other non-date index levels identify independent copies of the
    hierarchy (one per entity of a panel): each copy is reconciled on its own,
    with its own in-sample statistics. A time series is a single copy.

    Reconcilers that learn from in-sample errors (``MinTrace`` with
    ``'mint_shrink'``, ``'mint_cov'``, ``'wls_var'``; ``ERM``; ``TopDown`` with
    average proportions) read the training rows of ``X`` and ``y``: the base
    forecasts of ``X`` at fit must then be out-of-sample forecasts, produced
    fold by fold with a ``tsforecast`` splitter (each fold predicting its test
    rows, for instance from ``cross_validate(..., return_estimator=True,
    return_indices=True)``; ``cross_val_predict`` refuses splitters whose test
    folds do not cover every row), never in-sample fitted values, which would
    understate the errors.

    The reconciler passed to the constructor is never modified: each
    reconciliation works on a copy. ``GridSearchCV`` compares reconcilers with
    ``param_grid={'reconciler': [BottomUp(), MinTrace(method='ols')]}``.

    Constraints, each raising an explicit ``ValueError``: every date to
    predict holds a row for every node of its copy; a node or a level-based
    copy unknown at fit cannot be predicted (a copy of a mapping hierarchy can,
    unless the reconciler needs in-sample data); base forecasts to reconcile
    are finite; top-down and middle-out reconcilers need a strictly
    hierarchical structure whose leaves all lie on the finest level.

    With ``temporal_hierarchy`` (``{'year': 12, 'quarter': 3, 'month': 1}``:
    number of fine steps of a node of each level), the index level
    ``temporal_level`` holds the temporal level of each row and the
    reconciliation is temporal (``hierarchy=None``: every other non-date
    level identifies a series reconciled on its own) or cross-temporal (with a
    ``hierarchy``: the summing matrix is the Kronecker product of both, so that
    the forecasts become coherent across the nodes and across the temporal
    levels at once). The fine grid is the one of the rows of the finest level;
    a period of the most aggregated level is that grid multiplied by its number
    of steps, aligned on the Unix epoch (calendar years for monthly and
    quarterly grids, days for hourly grids) or on ``temporal_origin``. An
    aggregate row is dated by the first fine step it covers on a grid labelling
    its periods by their start (``'MS'``, ``'QS'``, ``'D'``, ``'h'``), by the last
    one on a grid labelling them by their end (``'ME'``, ``'QE'``, ``'W-SUN'``).
    Each period of ``X`` is reconciled on its own and must be complete at
    predict (every node of every temporal level): test sets cover complete
    periods, as the folds of ``period_splits``. Reconcilers and intervals
    reading in-sample errors are refused. Rows indexed by forecast origin
    aggregate origins: with a horizon of ``h`` fine steps, periods of origins
    starting ``h`` steps before the calendar periods of the targets
    (``temporal_origin``) reconcile calendar periods of the targets.

    Args:
        hierarchy: List of lists of index level names, or ``{parent:
            [children]}`` mapping on the labels of ``node_level``.
        reconciler: hierarchicalforecast reconciler (``BottomUp()``,
            ``MinTrace(method='ols')``...); None for ``BottomUp()``.
        node_level: For a mapping: name or position of the index level
            carrying the nodes; None for the level just before the date.
        aggregate_label: For a list of levels: label of the levels an
            aggregate row aggregates.
        model_combination: How the base forecast columns become one forecast:
            None (a single column), ``'mean'`` (mean of the columns) or a
            ``{level: column}`` mapping covering every level of the hierarchy
            (one base model per level; other columns ignored).
        forecast_columns: Base forecast columns of ``X``; None for every
            column but the interval columns ``'<column>-lo-<level>'`` and
            ``'<column>-hi-<level>'``. Other columns (exogenous variables)
            play no part in the reconciliation.
        temporal_hierarchy: Temporal hierarchy ``{name: number of fine
            steps}`` (one level of 1 step; every count divides the largest);
            None for a cross-sectional reconciliation.
        temporal_level: With a temporal hierarchy: name or position of the
            non-date index level holding the temporal level of each row; None
            for the level just before the date (``node_level`` then defaults
            to the level before it).
        temporal_origin: With a temporal hierarchy: a date at which a period
            of the most aggregated level starts (fiscal years, weeks starting
            on Monday, periods of origins); None for the Unix epoch.

    Attributes:
        reconciler_: Copy of the reconciler used by ``predict``.
        hierarchy_levels_: Names of the levels of the hierarchy, most
            aggregated first (keys of a ``model_combination`` mapping).
        copies_: Keys of the copies of the hierarchy seen at fit (``((),)``
            without copy levels).
        forecast_columns_: Base forecast columns of ``X``.
        temporal_freq_: Pandas alias of the fine grid of a temporal
            hierarchy, None without one.
        n_features_in_: Number of columns of ``X`` seen at fit.
        feature_names_in_: Column names of ``X`` seen at fit.

    Examples:
        >>> import pandas as pd
        >>> from hierarchicalforecast.methods import BottomUp
        >>> from tsforecast.adapters import HierarchicalForecastAdapter
        >>> dates = pd.date_range('2024-01-01', periods=2, freq='MS', name='date')
        >>> index = pd.MultiIndex.from_product([['a', 'b', 'c'], dates], names=['node', 'date'])
        >>> y = pd.Series([3.0, 5.0, 1.0, 2.0, 2.0, 3.0], index=index)
        >>> X = pd.DataFrame({'model': [3.5, 4.0, 1.0, 2.5, 2.0, 3.0]}, index=index)
        >>> adapter = HierarchicalForecastAdapter({'a': ['b', 'c']}, BottomUp()).fit(X, y)
        >>> adapter.predict(X)
        array([3. , 5.5, 1. , 2.5, 2. , 3. ])
        >>> adapter.reconcile(X).columns.tolist()
        ['model/BottomUp']

        Temporal reconciliation of four quarters and their year (dated by its
        first quarter):

        >>> from hierarchicalforecast.methods import MinTrace
        >>> quarters = pd.date_range('2024-01-01', periods=4, freq='QS', name='date')
        >>> index = pd.MultiIndex.from_arrays(
        ...     [['quarter'] * 4 + ['year'], quarters.append(quarters[:1])], names=['level', 'date'])
        >>> X = pd.DataFrame({'model': [2.0, 3.0, 2.0, 3.0, 12.0]}, index=index)
        >>> adapter = HierarchicalForecastAdapter(
        ...     None, MinTrace(method='ols'), temporal_hierarchy={'year': 4, 'quarter': 1}).fit(X, X['model'])
        >>> adapter.predict(X).round(2)
        array([ 2.4,  3.4,  2.4,  3.4, 11.6])
    """

    # Initialisation
    def __init__(
        self,
        hierarchy: Any = None,
        reconciler: Any = None,
        *,
        node_level: Any = None,
        aggregate_label: Any = "Total",
        model_combination: Any = None,
        forecast_columns: Any = None,
        temporal_hierarchy: Optional[Mapping] = None,
        temporal_level: Any = None,
        temporal_origin: Any = None,
    ):
        # Stockage des paramètres tels quels (convention sklearn : validation au fit)
        self.hierarchy = hierarchy
        self.reconciler = reconciler
        self.node_level = node_level
        self.aggregate_label = aggregate_label
        self.model_combination = model_combination
        self.forecast_columns = forecast_columns
        self.temporal_hierarchy = temporal_hierarchy
        self.temporal_level = temporal_level
        self.temporal_origin = temporal_origin

    # ------------------------------------------------------------------
    # Méthodes auxiliaires
    # ------------------------------------------------------------------

    # Vérification d'un réconciliateur
    @staticmethod
    def _check_reconciler(
        reconciler: Any, structures: Sequence[Structure], level_names: Sequence[str], temporal: bool = False
    ) -> None:
        """Refuse an object that is no reconciler, or a reconciler the structure or the mode does not allow.

        Args:
            reconciler: Object to check.
            structures: Structures of the copies to reconcile.
            level_names: Names of the levels of the hierarchy.
            temporal: Whether the reconciliation is temporal or cross-temporal.

        Raises:
            TypeError: If ``reconciler`` is not a hierarchicalforecast reconciler.
            ValueError: If the reconciler learns from in-sample errors in a
                temporal mode, needs a strictly hierarchical structure that a
                copy lacks, or names a middle level absent from the hierarchy.
        """
        # Import des méthodes de réconciliation et des utilitaires
        _, methods, utils = _import_hierarchicalforecast()
        # Vérification du type du réconciliateur
        if not isinstance(reconciler, methods.HReconciler):
            raise TypeError(
                f"reconciler must be a hierarchicalforecast reconciler (BottomUp(), MinTrace(...)...), "
                f"got {type(reconciler).__name__}"
            )
        # Nom de la classe, pour les messages d'erreur
        name = type(reconciler).__name__
        # Modes temporels : aucune erreur insample par période n'est disponible
        if temporal and reconciler.insample:
            raise ValueError(
                f"{name} learns from in-sample errors, which temporal reconciliation does not support: use "
                "BottomUp, MinTrace with method 'ols' or 'wls_struct', or TopDown with 'forecast_proportions'"
            )
        # Réconciliateurs qui descendent les niveaux : hiérarchie stricte, feuilles au niveau le plus fin
        if getattr(reconciler, "is_strictly_hierarchical", False):
            # Parcours des structures : une par copie
            for structure in structures:
                # Hiérarchie à un seul niveau : aucune descente à effectuer
                if len(structure.levels) < 2:
                    continue
                # Contrôle de la position des feuilles, puis de l'unicité du parent de chaque nœud
                if not structure.bottom_is_finest_level() or not utils.is_strictly_hierarchical(
                    structure.summing, structure.level_positions()
                ):
                    raise ValueError(
                        f"{name} needs a strictly hierarchical structure (one parent per node) whose leaves "
                        "all lie on the finest level; use BottomUp or MinTrace for this hierarchy"
                    )
        # Niveau intermédiaire demandé par un réconciliateur 'middle-out'
        middle_level = getattr(reconciler, "middle_level", None)
        if middle_level is not None and middle_level not in level_names:
            raise ValueError(
                f"{name} middle_level {middle_level!r} is not a level of the hierarchy {list(level_names)}"
            )

    # Liste des réconciliateurs d'un appel
    def _resolve_reconcilers(self, reconcilers: Any) -> List[Any]:
        """Return the reconcilers of a call, checked against the fitted structures.

        Args:
            reconcilers: A reconciler, a list of reconcilers, or None for the
                fitted ``reconciler_``.

        Returns:
            The list of reconcilers to run.

        Raises:
            TypeError: If an object is not a hierarchicalforecast reconciler.
            ValueError: If the list is empty or a reconciler does not suit the
                fitted structures or mode.
        """
        # Par défaut : le réconciliateur ajusté
        if reconcilers is None:
            return [self.reconciler_]
        # Conversion en liste d'un réconciliateur isolé
        reconcilers = list(reconcilers) if isinstance(reconcilers, (list, tuple)) else [reconcilers]
        # Refus d'une liste vide
        if not reconcilers:
            raise ValueError("reconcilers must hold at least one reconciler")
        # Structures des copies ajustées
        structures = [record.structure for record in self._copies.values()]
        # Vérification de chaque réconciliateur
        for reconciler in reconcilers:
            self._check_reconciler(
                reconciler, structures, self.hierarchy_levels_, self._design.mode != _CROSS_SECTION
            )
        return reconcilers

    # Localisation des lignes d'un jeu de données à prédire
    def _locate_rows(self, layout: TemporalLayout, needs_insample: bool) -> _RowNodes:
        """Locate the copy, the node and the reconciliation column of each row of ``X``.

        Args:
            layout: Temporal and entity structure of the rows of ``X``.
            needs_insample: Whether the reconciliation reads the training rows
                of each copy.

        Returns:
            The copy, the node, the level of the node and the reconciliation
            column of each row.

        Raises:
            ValueError: If a copy is unknown at fit while its nodes or its
                training rows are needed, or if a row is off its structure.
        """
        # Copies, nœuds et niveaux temporels des combinaisons de niveaux d'index de X
        split = _split_entities(layout, self._design)
        # Données structurelles de chaque copie de X
        records = []
        # Copies de X inexploitables
        unknown_copies = []
        # Parcours des copies de X : copie ajustée, ou structure commune à toute copie
        for copy_key in split.copy_keys:
            record = self._copies.get(copy_key)
            if record is None and self._shared is not None and not needs_insample:
                # Copie inconnue dont la structure est connue sans données d'entraînement
                record = self._shared
            if record is None:
                # Cas d'une copie inexploitable : signalée à la fin du parcours
                unknown_copies.append(copy_key)
            records.append(record)
        # Refus des copies inexploitables, avec la raison propre à la forme de la hiérarchie
        if unknown_copies:
            reason = (
                "the reconciliation needs their training rows"
                if self._shared is not None
                else "their nodes are learnt from the training rows"
            )
            raise ValueError(f"Copies of the hierarchy unknown at fit ({reason}): {_preview(unknown_copies)}")
        # Nœud et colonne de chaque ligne
        return _place_rows(layout, self._design, split, records)

    # Prévisions de base des modèles d'un appel
    @staticmethod
    def _models(
        X: pd.DataFrame, columns: Sequence[Hashable], combination: Any, row_level: np.ndarray, which: str
    ) -> List[Tuple[Hashable, str, np.ndarray]]:
        """Return ``(name, internal name, values)`` of the base models of a call.

        ``which='target'``: the forecast that ``predict`` reconciles (the
        combined forecast, or the single base column); ``'all'``: every base
        column, then the combined forecast.

        Args:
            X: Base forecasts of the rows.
            columns: Base forecast columns.
            combination: Resolved combination: None, ``'mean'`` or the column
                position of each level.
            row_level: Position of the level of the node of each row.
            which: ``'target'`` or ``'all'``.

        Returns:
            Triples ``(name, internal name, values)``, one per model, the
            values aligned with the rows of ``X``.
        """
        # Matrice des prévisions de base (lignes × modèles)
        base = X[list(columns)].to_numpy(dtype=np.float64)
        # Liste des triplets (nom public, nom interne, valeurs)
        models = []
        # Prévisions de base : toutes pour 'all', la seule colonne en l'absence de combinaison
        if which == "all" or combination is None:
            # Nombre de colonnes retenues
            n_base = base.shape[1] if which == "all" else 1
            models = [(columns[i], f"{_MODEL_PREFIX}{i}", base[:, i]) for i in range(n_base)]
        # Ajout de la prévision combinée
        if combination is not None:
            if isinstance(combination, str):
                # Moyenne des modèles, ligne par ligne
                combined = base.mean(axis=1)
            else:
                # Un modèle de base par niveau de la hiérarchie
                combined = base[np.arange(len(base)), combination[row_level]]
            models.append((COMBINED_NAME, _COMBINED_INTERNAL, combined))
        return models

    # Colonnes d'intervalle des modèles pour les intervalles par loi normale
    def _paired_levels(self, column: Hashable) -> List[str]:
        """Levels for which ``X`` holds both interval columns of a base column.

        Args:
            column: Base forecast column.

        Returns:
            The confidence levels (as written in the column names) with both a
            ``'-lo-'`` and a ``'-hi-'`` column for ``column``.
        """
        # Niveaux de confiance disposant des deux bornes
        return [
            level
            for (model, side, level) in self._intervals
            if model == column and side == "lo" and (model, "hi", level) in self._intervals
        ]

    # Construction des colonnes d'intervalle au format de hierarchicalforecast
    def _interval_values(
        self, X: pd.DataFrame, models: Sequence[tuple], row_level: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Interval columns of the base models, named after their internal names.

        Args:
            X: Base forecasts and interval columns of the rows.
            models: Triples ``(name, internal name, values)`` of the reconciled models.
            row_level: Position of the level of the node of each row.

        Returns:
            Dictionary ``'<internal>-<side>-<level>'`` -> interval bound of
            each row; a combination by level takes the bounds of the column
            chosen for the level of each row.

        Raises:
            ValueError: If a base model lacks its interval columns, if the
                combined forecast is a mean of models, or if the columns chosen
                by the combination share no confidence level.
        """
        # Dictionnaire nom interne de la colonne d'intervalle -> valeurs
        values = {}
        # Parcours des modèles réconciliés
        for name, internal, _ in models:
            # Cas d'un modèle de base : ses propres colonnes d'intervalle
            if internal != _COMBINED_INTERNAL:
                # Dictionnaire niveau de confiance -> colonne encadrée
                sources = {level: name for level in self._paired_levels(name)}
                if not sources:
                    raise ValueError(
                        f"intervals_method='normality' needs the interval columns '{name}-lo-<level>' and "
                        f"'{name}-hi-<level>' in X"
                    )
                # Reprise des deux bornes de chaque niveau sous le nom interne
                for level, column in sources.items():
                    for side in _INTERVAL_SIDES:
                        values[f"{internal}-{side}-{level}"] = X[self._intervals[(column, side, level)]].to_numpy(
                            dtype=np.float64
                        )
                continue
            # Cas d'une moyenne : aucun intervalle déductible de ceux des modèles
            if isinstance(self._combination, str):
                raise ValueError(
                    "intervals_method='normality' needs prediction intervals of the base forecasts, and a mean "
                    "of models has none: use intervals_method='bootstrap' or 'conformal'"
                )
            # Colonne choisie pour chaque niveau de la hiérarchie
            chosen = [self.forecast_columns_[position] for position in self._combination]
            # Colonnes choisies, sans répétition
            distinct = list(dict.fromkeys(chosen))
            # Niveaux de confiance communs à toutes les colonnes choisies
            common = set.intersection(*(set(self._paired_levels(column)) for column in distinct))
            if not common:
                raise ValueError(
                    "intervals_method='normality' needs, for a common level, the interval columns of every "
                    f"base column chosen by model_combination: {_preview(distinct)}"
                )
            # Pour chaque niveau de la hiérarchie, position de sa colonne parmi les colonnes distinctes
            column_of_level = np.array([distinct.index(column) for column in chosen], dtype=np.intp)
            # Parcours des niveaux de confiance, en ordre croissant, et de leurs deux bornes
            for level in sorted(common, key=float):
                for side in _INTERVAL_SIDES:
                    # Matrice d'une borne (lignes × colonnes distinctes)
                    matrix = np.column_stack(
                        [X[self._intervals[(column, side, level)]].to_numpy(dtype=np.float64) for column in distinct]
                    )
                    # Sélection, sur chaque ligne, de la borne du modèle retenu pour son niveau
                    values[f"{internal}-{side}-{level}"] = matrix[np.arange(len(matrix)), column_of_level[row_level]]
        return values

    # Réconciliation des lignes d'un jeu de données, copie par copie
    def _run(
        self,
        X: pd.DataFrame,
        reconcilers: Sequence[Any],
        which: str,
        *,
        level: Optional[Sequence[float]] = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        seed: int = 0,
        diagnostics: bool = False,
        atol: float = 1e-6,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[Tuple[tuple, pd.DataFrame]]]:
        """Reconcile the rows of ``X``, copy by copy.

        Args:
            X: Base forecasts of the rows to reconcile.
            reconcilers: Reconcilers to run.
            which: ``'target'`` for the forecast that ``predict`` reconciles,
                ``'all'`` for every base model and the combined forecast.
            level: Confidence levels of the prediction intervals, or None.
            intervals_method: Method of the prediction intervals
                (``'normality'``, ``'bootstrap'``, ``'permbu'``, ``'conformal'``).
            num_samples: Number of samples drawn by sampling methods.
            seed: Seed of the sampling methods.
            diagnostics: Whether to compute the coherence diagnostics of each copy.
            atol: Absolute tolerance of the coherence diagnostics.

        Returns:
            Quadruple ``(X, values, names, diagnostics)``: the validated ``X``,
            the reconciled values (one row per row of ``X``, one column per
            output), the public name of each output column, and the pairs
            ``(copy key, diagnostics frame)``.

        Raises:
            ValueError: If the intervals need in-sample errors in a temporal
                mode, if a base forecast is missing or infinite, if a node has no
                training row while the reconciliation needs them, or if the
                bootstrap has too few training dates.
        """
        # Import différé du moteur de réconciliation
        core, _, _ = _import_hierarchicalforecast()
        # Vérification de X et analyse de son index (aucune grille exigée)
        X, _, layout = check_adapter_input(self, X, reset=False, detect_freq=False)
        # Cas d'un jeu vide : aucune réconciliation
        if len(X) == 0:
            return X, np.empty((0, 0)), [], []
        # Besoin des lignes d'entraînement : réconciliateur ou méthode d'intervalle insample
        needs_insample = any(reconciler.insample for reconciler in reconcilers) or (
            level is not None and intervals_method in _INSAMPLE_INTERVAL_METHODS
        )
        # Modes temporels : aucune erreur insample par période n'est disponible
        temporal = self._design.mode != _CROSS_SECTION
        if temporal and level is not None and intervals_method in _INSAMPLE_INTERVAL_METHODS:
            raise ValueError(
                f"intervals_method={intervals_method!r} reads in-sample errors, which temporal reconciliation does "
                "not support: use intervals_method='normality'"
            )
        # Copie, nœud, niveau et colonne de chaque ligne
        rows = self._locate_rows(layout, needs_insample)
        # Prévisions de base à réconcilier
        models = self._models(X, self.forecast_columns_, self._combination, rows.row_level, which)
        # Vérification que toutes les prévisions à réconcilier sont finies
        for name, _, values in models:
            invalid = ~np.isfinite(values)
            if invalid.any():
                raise ValueError(
                    f"The base forecast {name!r} is missing or infinite on {int(invalid.sum())} row(s) of X: "
                    "every row to reconcile needs a finite base forecast"
                )
        # Colonnes d'intervalle, lues par la seule méthode par loi normale
        intervals = (
            self._interval_values(X, models, rows.row_level)
            if level is not None and intervals_method == "normality"
            else {}
        )
        # Dictionnaire nom interne -> nom public des modèles
        names = {internal: name for name, internal, _ in models}

        # Initialisation des résultats : valeurs (lignes de X × colonnes de sortie), noms de ces colonnes
        # et diagnostics de chaque copie
        output, columns, per_copy = None, [], []
        # Parcours des copies : une réconciliation indépendante par copie
        for code, (copy_key, structure) in enumerate(zip(rows.copy_keys, rows.structures)):
            # Lignes de X de la copie, avec leurs nœuds et leurs colonnes (dates ou débuts de période)
            members = np.flatnonzero(rows.row_copy == code)
            nodes = rows.row_node[members]
            dates = rows.row_column[members]
            # Vérification de la présence de tous les nœuds dans chaque colonne
            self._check_complete(structure, nodes, dates, copy_key, temporal)
            # Construction des prévisions de base au format de hierarchicalforecast : identifiant, colonne
            # (date ou début de période), modèles, puis intervalles
            frame = {_ID_COL: structure.ids[nodes], _TIME_COL: dates}
            frame.update({internal: values[members] for _, internal, values in models})
            frame.update({column: values[members] for column, values in intervals.items()})
            Y_hat_df = pd.DataFrame(frame)
            # Lignes d'entraînement et équilibre de leur grille, absents sans besoin insample
            insample, balanced = None, False
            # Ajout des lignes d'entraînement de la copie
            if needs_insample:
                # Données ajustées de la copie
                record = self._copies[copy_key]
                # Vérification que chaque nœud dispose de lignes d'entraînement
                if record.missing_nodes:
                    where = f" of the copy {copy_key!r}" if copy_key else ""
                    raise ValueError(
                        f"The reconciliation needs training rows of every node, but the nodes{where} "
                        f"{_preview(record.missing_nodes)} have none"
                    )
                # Restriction aux modèles réconciliés
                insample = record.insample[[_ID_COL, _TIME_COL, _TARGET_COL, *names]]
                balanced = record.balanced
                # Bootstrap : blocs d'autant de dates d'entraînement consécutives que de dates à réconcilier
                n_train, n_dates = insample[_TIME_COL].nunique(), len(dates.unique())
                if level is not None and intervals_method == "bootstrap" and n_train <= n_dates:
                    raise ValueError(
                        f"intervals_method='bootstrap' draws blocks of {n_dates} consecutive training dates (as "
                        f"many as the dates to reconcile), but the training rows cover {n_train} dates: "
                        "reconcile fewer dates at once"
                    )

            # Objet local et copies des réconciliateurs : reconcile modifie les réconciliateurs et trie level
            reconciliation = core.HierarchicalReconciliation(reconcilers=[copy.deepcopy(r) for r in reconcilers])
            # Réconciliation de la copie
            result = reconciliation.reconcile(
                Y_hat_df=Y_hat_df,
                tags=structure.tags(),
                S_df=structure.summing_frame(),
                Y_df=insample,
                level=None if level is None else list(level),
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                is_balanced=balanced,
                diagnostics=diagnostics,
                diagnostics_atol=atol,
            )
            # Initialisation du résultat à la première copie : colonnes ajoutées par la réconciliation
            if output is None:
                columns = [column for column in result.columns if column not in Y_hat_df.columns]
                output = np.full((len(X), len(columns)), np.nan)
            # Sortie triée par nœud puis colonne : réalignement par clés sur les lignes de X
            source = pd.MultiIndex.from_arrays([result[_ID_COL].to_numpy(), pd.DatetimeIndex(result[_TIME_COL])])
            target = pd.MultiIndex.from_arrays([Y_hat_df[_ID_COL].to_numpy(), pd.DatetimeIndex(dates)])
            # Écriture des valeurs aux positions des lignes de la copie
            output[members] = restore_row_order(result[columns].to_numpy(dtype=np.float64), source, target)
            # Conservation des diagnostics de la copie
            if diagnostics:
                per_copy.append((copy_key, reconciliation.diagnostics))
        # Retour des résultats, les colonnes portant le nom des modèles de l'utilisateur
        return X, output, [self._public_name(column, names) for column in columns], per_copy

    # Nom public d'une colonne de hierarchicalforecast
    @staticmethod
    def _public_name(column: str, names: Mapping[str, Hashable]) -> str:
        """Replace the internal model name that prefixes a column with the name of the base model.

        Args:
            column: Output column of hierarchicalforecast (``'<internal>/<Reconciler>...'``).
            names: Dictionary internal model name -> base model name.

        Returns:
            The column name with its prefix replaced by the base model name.
        """
        # Découpage du nom '<modèle interne>/<réconciliateur...>'
        internal, separator, rest = column.partition("/")
        # Nom inchangé pour un modèle interne inconnu
        return f"{names.get(internal, internal)}{separator}{rest}"

    # Contrôle de la présence de tous les nœuds dans chaque colonne
    @staticmethod
    def _check_complete(
        structure: Structure, nodes: np.ndarray, dates: pd.DatetimeIndex, copy_key: tuple, temporal: bool = False
    ) -> None:
        """Refuse a date, or a period of the most aggregated temporal level, that lacks a node of the copy.

        Args:
            structure: Structure of the copy.
            nodes: Node position of each row of the copy.
            dates: Reconciliation column of each row (date, or start of period).
            copy_key: Key of the copy, for the error message.
            temporal: Whether the columns are periods of a temporal hierarchy.

        Raises:
            ValueError: If a date or a period has no row for some node of the copy.
        """
        # Code de la colonne de chaque ligne et calendrier des colonnes distinctes
        date_codes, calendar = pd.factorize(dates)
        # Nombre de lignes par date
        counts = np.bincount(date_codes, minlength=len(calendar))
        # Dates dont le nombre de lignes diffère du nombre de nœuds (couples nœud-date uniques)
        incomplete = np.flatnonzero(counts != structure.n_nodes)
        # Cas d'au moins une date incomplète : erreur décrivant la première
        if incomplete.size:
            first = incomplete[0]
            # Nœuds présents à cette date, puis nœuds manquants
            present = nodes[date_codes == first]
            missing = [structure.node_keys[node] for node in np.setdiff1d(np.arange(structure.n_nodes), present)]
            where = f" of the copy {copy_key!r}" if copy_key else ""
            # Cas temporel : période incomplète du niveau le plus agrégé
            if temporal:
                raise ValueError(
                    f"X has no row for the nodes{where} {_preview(missing)} in the period starting {calendar[first]} "
                    f"({incomplete.size} incomplete period(s)): each period of the most aggregated temporal level "
                    "to reconcile needs the base forecast of every node of every temporal level; test sets must "
                    "cover complete periods (see period_splits)"
                )
            raise ValueError(
                f"X has no row for the nodes{where} {_preview(missing)} on {calendar[first]} "
                f"({incomplete.size} incomplete date(s)): each date to reconcile needs the base forecast of "
                "every node of the hierarchy"
            )

    # ------------------------------------------------------------------
    # API sklearn
    # ------------------------------------------------------------------

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Any) -> "HierarchicalForecastAdapter":
        """Build the structure of each copy of the hierarchy and keep the training rows.

        Args:
            X: Base forecasts of the training rows (out-of-sample forecasts for
                the reconcilers that learn from in-sample errors), indexed by
                ``(copy..., node..., date)`` (``(copy..., node..., temporal
                level, date)`` with a temporal hierarchy); rows in any order.
            y: Realised values of the same rows (every node, aggregates
                included).

        Returns:
            The fitted adapter.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            TypeError: If ``X`` is not a ``DataFrame``, if its dates are not
                ``datetime64``, or if ``reconciler`` is not a hierarchicalforecast
                reconciler.
            ValueError: If ``hierarchy`` is missing or malformed, if a row
                matches no node of the hierarchy, if ``y`` is not a single
                numeric target aligned with ``X``, if the base forecast columns
                or ``model_combination`` are invalid, or if the reconciler does
                not suit the hierarchy or lacks training rows of a node; with a
                temporal hierarchy, if it is malformed, if a row holds an
                unknown temporal level or is misdated, if the fine grid cannot
                be detected or does not tile the periods, or if the reconciler
                learns from in-sample errors.
        """
        # Import différé des méthodes de réconciliation
        _, methods, _ = _import_hierarchicalforecast()
        # Vérification des paramètres indispensables
        if self.temporal_hierarchy is None:
            # Paramètres temporels sans hiérarchie temporelle
            if self.temporal_level is not None or self.temporal_origin is not None:
                raise ValueError("temporal_level and temporal_origin only apply with a temporal_hierarchy")
            if self.hierarchy is None:
                raise ValueError(
                    "hierarchy is required: a list of lists of index levels or a {parent: [children]} mapping "
                    "(or set temporal_hierarchy for a temporal reconciliation)"
                )
        # Vérification des entrées et analyse de l'index (aucune grille exigée)
        X, y, layout = check_adapter_input(self, X, y, reset=True, detect_freq=False)
        # Vérification d'une cible unique
        if y is None or y.ndim != 1:
            raise ValueError("y must be a single target: a Series indexed like X")
        # Valeurs de la cible, alignées sur les lignes de X
        target = np.asarray(y, dtype=np.float64)

        # Mode, hiérarchies et calendrier, depuis les niveaux d'index
        design = self._build_design(layout)
        # Colonnes des prévisions de base et de leurs intervalles
        base, intervals = _resolve_columns(X, self.forecast_columns)
        # Mode de combinaison des modèles de base
        combination = _resolve_combination(self.model_combination, design.level_names, base)
        # Copie du réconciliateur (BottomUp par défaut) : l'objet reçu n'est jamais modifié
        reconciler = methods.BottomUp() if self.reconciler is None else copy.deepcopy(self.reconciler)

        # Copies, nœuds et niveaux temporels des combinaisons de niveaux d'index
        split = _split_entities(layout, design)
        # Structures de chaque copie, dans l'ordre de leurs codes
        records = []
        # Parcours des copies : structure depuis leurs nœuds observés (structure temporelle commune sinon)
        for code in range(len(split.copy_keys)):
            # Clés de nœud cross-section distinctes des combinaisons de niveaux d'index de la copie
            members = np.flatnonzero(split.entity_copy == code)
            keys = list(dict.fromkeys(split.node_keys[member] for member in members))
            records.append(design.record(None if design.spec is None else design.spec.build_structure(keys)))
        # Vérification de la compatibilité du réconciliateur avec les structures et le mode
        self._check_reconciler(
            reconciler, [record.structure for record in records], design.level_names, design.mode != _CROSS_SECTION
        )
        # Nœud, niveau et colonne de chaque ligne
        rows = _place_rows(layout, design, split, records)
        # Toutes les prévisions de base, combinaison comprise
        models = self._models(X, base, combination, rows.row_level, "all")

        # Dictionnaire clé de copie -> données ajustées de la copie
        copies = {}
        # Parcours des copies : structures, plus lignes d'entraînement en cross-section
        for code, (copy_key, record) in enumerate(zip(split.copy_keys, records)):
            # Modes temporels : aucune donnée insample, seules les structures sont conservées
            if design.mode != _CROSS_SECTION:
                copies[copy_key] = record
                continue
            structure = record.structure
            # Lignes de la copie dont la cible est renseignée
            members = np.flatnonzero((rows.row_copy == code) & np.isfinite(target))
            # Ordre canonique (nœud, date) : l'ajustement ne dépend pas de l'ordre des lignes
            members = members[np.lexsort((layout.dates[members].to_numpy(), rows.row_node[members]))]
            # Lignes d'entraînement au format de hierarchicalforecast
            frame = {
                _ID_COL: structure.ids[rows.row_node[members]],
                _TIME_COL: layout.dates[members],
                _TARGET_COL: target[members],
            }
            frame.update({internal: values[members] for _, internal, values in models})
            # Indicatrice des nœuds disposant d'au moins une ligne
            covered = np.zeros(structure.n_nodes, dtype=bool)
            covered[rows.row_node[members]] = True
            # Nombre de dates d'entraînement de la copie
            n_dates = len(layout.dates[members].unique())
            # Enregistrement des données ajustées (grille équilibrée : chaque nœud présent à chaque date)
            copies[copy_key] = _CopyFit(
                structure=structure,
                cross_section=structure,
                insample=pd.DataFrame(frame),
                missing_nodes=tuple(structure.node_keys[node] for node in np.flatnonzero(~covered)),
                balanced=len(members) == structure.n_nodes * n_dates,
            )
        # Réconciliateur insample : lignes d'entraînement exigées pour chaque nœud
        if reconciler.insample:
            # Parcours des copies : refus d'un nœud sans ligne
            for copy_key, record in copies.items():
                if record.missing_nodes:
                    where = f" of the copy {copy_key!r}" if copy_key else ""
                    raise ValueError(
                        f"{type(reconciler).__name__} learns from in-sample errors, but the nodes{where} "
                        f"{_preview(record.missing_nodes)} have no training row with a target"
                    )

        # Attributs ajustés, posés une fois les contrôles réussis
        self._design = design
        # Structures de toute copie inconnue au fit : arbre cross-section et/ou structure temporelle ;
        # None quand les nœuds d'une copie s'apprennent de ses lignes
        self._shared = (
            design.record(None if design.spec is None else design.spec.tree)
            if design.spec is None or design.spec.kind == "tree"
            else None
        )
        self._intervals = intervals
        self._combination = combination
        self._copies = copies
        self.hierarchy_levels_ = design.level_names
        self.forecast_columns_ = base
        self.copies_ = split.copy_keys
        self.temporal_freq_ = None if design.calendar is None else design.calendar.freq
        self.reconciler_ = reconciler
        return self

    # Construction du mode de réconciliation depuis les niveaux d'index
    def _build_design(self, layout: TemporalLayout) -> _Design:
        """Read the hierarchies against the index levels and choose the mode of reconciliation.

        Args:
            layout: Temporal and entity structure of the training rows.

        Returns:
            The design: cross-sectional mode without ``temporal_hierarchy``,
            temporal mode without ``hierarchy``, cross-temporal mode with both.

        Raises:
            ValueError: If a hierarchy does not match the index levels, or if
                ``node_level`` is given without ``hierarchy``.
        """
        # Noms des niveaux d'index précédant la date
        names = list(layout.entity_names)
        # Mode cross-section : hiérarchie lue sur tous les niveaux d'index
        if self.temporal_hierarchy is None:
            spec = parse_hierarchy(self.hierarchy, [*names, layout.time_name], self.node_level, self.aggregate_label)
            return _Design(
                mode=_CROSS_SECTION,
                spec=spec,
                level_names=spec.level_names,
                copy_names=_copy_names(spec.copy_names, spec.copy_levels),
            )
        # Hiérarchie temporelle et niveau temporel de chaque ligne
        temporal = parse_temporal_hierarchy(self.temporal_hierarchy)
        temporal_index = read_temporal_index(layout, temporal, self.temporal_level)
        # Grille fine détectée sur les lignes du niveau le plus fin, et calendrier des périodes
        calendar = TemporalCalendar.build(
            temporal, detect_fine_frequency(layout, temporal_index.row_levels, temporal), self.temporal_origin
        )
        # Noms des niveaux d'index autres que le niveau temporel
        others = [name for position, name in enumerate(names) if position != temporal_index.position]
        # Paramètres communs aux deux modes temporels
        common = dict(
            temporal=temporal,
            temporal_level=self.temporal_level,
            temporal_name=names[temporal_index.position],
            calendar=calendar,
            temporal_structure=temporal_structure(temporal),
        )
        # Mode temporel : chaque combinaison des autres niveaux est une copie
        if self.hierarchy is None:
            if self.node_level is not None:
                raise ValueError("node_level only applies with a hierarchy")
            return _Design(
                mode=_TEMPORAL,
                spec=None,
                level_names=temporal.levels,
                copy_names=_copy_names(tuple(others), tuple(range(len(others)))),
                **common,
            )
        # Mode cross-temporel : hiérarchie lue sur les autres niveaux, niveaux croisés
        spec = parse_hierarchy(self.hierarchy, [*others, layout.time_name], self.node_level, self.aggregate_label)
        return _Design(
            mode=_CROSS_TEMPORAL,
            spec=spec,
            level_names=tuple(
                f"{cs_level}{CROSS_LEVEL_SEPARATOR}{te_level}"
                for cs_level in spec.level_names
                for te_level in temporal.levels
            ),
            copy_names=_copy_names(spec.copy_names, spec.copy_levels),
            **common,
        )

    # Méthode de prédiction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Reconcile the base forecasts of the rows of ``X`` with ``reconciler``.

        Args:
            X: Base forecasts with the columns seen at fit; on each date of each
                copy, one row per node of the hierarchy (with a temporal
                hierarchy: in each period of its most aggregated level, one row
                per node and per temporal node); rows in any order.

        Returns:
            Reconciled forecasts (mean), shape ``(n_samples,)``, in the row
            order of ``X``.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: If the columns differ from those seen at fit, if a node
                or a copy is unknown, if a date (or a period of a temporal
                hierarchy) lacks a node, if a temporal row is misdated, or if a
                base forecast is missing.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        # Réconciliation de la seule prévision cible
        _, values, _, _ = self._run(X, [self.reconciler_], "target")
        # Unique colonne de sortie (tableau vide pour un jeu vide)
        return values[:, 0].copy() if values.shape[1] else np.empty(0)

    # Méthode de réconciliation détaillée
    def reconcile(
        self,
        X: pd.DataFrame,
        reconcilers: Any = None,
        *,
        level: Optional[Sequence[float]] = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Reconcile every base model of ``X`` with one or several reconcilers.

        Args:
            X: Base forecasts, as for :meth:`predict`.
            reconcilers: A reconciler or a list of reconcilers; None for the
                fitted ``reconciler_``.
            level: Confidence levels of the prediction intervals (``[80,
                95]``), or None for point forecasts only.
            intervals_method: Method of the intervals: ``'normality'`` (reads
                the interval columns ``'<column>-lo-<level>'`` /
                ``'<column>-hi-<level>'`` of ``X``), ``'bootstrap'``,
                ``'permbu'`` or ``'conformal'`` (read the training rows; refused
                with a temporal hierarchy).
            num_samples: Number of coherent samples per row when positive
                (with ``level``).
            seed: Seed of the random generator of the intervals and samples.

        Returns:
            Frame indexed like ``X``, with one column ``'<model>/<reconciler>'``
            per base column (and ``'combined/<reconciler>'`` for a
            ``model_combination``) and per reconciler, plus ``'-lo-<level>'``,
            ``'-hi-<level>'`` and ``'-sample-<i>'`` columns when requested.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            NotFittedError: If the adapter is not fitted.
            TypeError: If a reconciler is not a hierarchicalforecast reconciler.
            ValueError: As :meth:`predict`, or if the intervals lack the
                columns or the training rows they read.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        # Réconciliateurs de l'appel
        reconcilers = self._resolve_reconcilers(reconcilers)
        # Réconciliation de toutes les prévisions de base
        X, values, names, _ = self._run(
            X, reconcilers, "all", level=level, intervals_method=intervals_method, num_samples=num_samples, seed=seed
        )
        # Tableau indexé comme X
        return pd.DataFrame(values, index=X.index, columns=pd.Index(names, dtype="str"))

    # Méthode de diagnostic de la cohérence
    def diagnose(self, X: pd.DataFrame, reconcilers: Any = None, *, atol: float = 1e-6) -> pd.DataFrame:
        """Measure the coherence of the base and reconciled forecasts of ``X``, level by level.

        Args:
            X: Base forecasts, as for :meth:`predict`.
            reconcilers: A reconciler or a list of reconcilers; None for the
                fitted ``reconciler_``.
            atol: Absolute tolerance of the coherence check.

        Returns:
            The diagnostics of hierarchicalforecast: columns ``level`` (levels
            of the hierarchy and ``'Overall'``), ``metric`` and one column per
            ``'<model>/<reconciler>'``; on a panel, the copy levels come first.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: As :meth:`reconcile`.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        # Réconciliateurs de l'appel
        reconcilers = self._resolve_reconcilers(reconcilers)
        # Réconciliation avec calcul des diagnostics
        X, _, _, per_copy = self._run(X, reconcilers, "all", diagnostics=True, atol=atol)
        # Dictionnaire nom interne -> nom public des modèles
        names = {f"{_MODEL_PREFIX}{i}": column for i, column in enumerate(self.forecast_columns_)}
        names[_COMBINED_INTERNAL] = COMBINED_NAME
        # Noms des colonnes des niveaux de copie (nom de substitution pour un niveau anonyme)
        copy_names = self._design.copy_names
        # Tableaux de diagnostics, un par copie
        pieces = []
        # Parcours des copies : renommage, typage et identification
        for copy_key, frame in per_copy:
            # Noms publics des modèles
            frame = frame.rename(columns=lambda column: self._public_name(column, names))
            # Colonnes de mesures en flottants, libellés en dtype chaîne
            measures = [column for column in frame.columns if column not in ("level", "metric")]
            frame = frame.astype({"level": "str", "metric": "str", **{column: np.float64 for column in measures}})
            # Insertion en tête des niveaux de copie, en ordre inverse pour conserver leur ordre
            for name, label in zip(reversed(copy_names), reversed(copy_key)):
                frame.insert(0, name, label)
            pieces.append(frame)
        # Cas d'un jeu vide : tableau sans ligne, réduit aux colonnes de libellés
        if not pieces:
            return pd.DataFrame(
                {"level": pd.Series(dtype="str"), "metric": pd.Series(dtype="str")}
            )
        # Concaténation des diagnostics des copies
        return pd.concat(pieces, ignore_index=True)

    # Matrice de sommation d'une copie
    def summing_matrix(self, copy: Any = None) -> pd.DataFrame:
        """Return the summing matrix of a copy of the hierarchy.

        Args:
            copy: Key of the copy (a label, or a tuple for several copy
                levels); None when the hierarchy has a single copy.

        Returns:
            Frame with one row per node (aggregates first) and one column per
            bottom node, labelled by the node labels (a ``MultiIndex`` of the
            hierarchy levels for a list of levels). With a temporal hierarchy,
            the matrix covers one period of its most aggregated level: temporal
            nodes are labelled ``'<level>-<rank in the period>'`` and, with a
            cross-sectional hierarchy too, each row is a pair (cross-sectional
            node, temporal node) of the Kronecker product of both matrices.

        Raises:
            NotFittedError: If the adapter is not fitted.
            ValueError: If ``copy`` is None with several copies, or unknown.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        # Détermination de la clé de la copie demandée
        if copy is None:
            # Copie par défaut : la seule copie ajustée
            if len(self.copies_) != 1:
                raise ValueError(f"Choose a copy among copies_: {_preview(self.copies_)}")
            key = self.copies_[0]
        else:
            # Clé fournie, normalisée en tuple
            key = copy if isinstance(copy, tuple) else (copy,)
        # Choix des structures : copie ajustée, ou structures communes pour une copie inconnue
        record = self._copies.get(key, self._shared)
        if record is None:
            raise ValueError(f"Copy {copy!r} unknown at fit; copies: {_preview(self.copies_)}")
        structure = record.structure
        # Mode et hiérarchie cross-section
        design, spec = self._design, self._design.spec
        # Étiquettes des nœuds selon le mode
        if design.mode == _TEMPORAL:
            # Temporel : nom du nœud temporel
            rows = pd.Index(list(structure.node_keys), name=design.temporal_name)
        else:
            # Clé cross-section (et nœud temporel) de chaque nœud, en tuple d'étiquettes
            keys = [key if design.mode == _CROSS_SECTION else key[0] for key in structure.node_keys]
            labels = [key if spec.kind == "levels" else (key,) for key in keys]
            names = list(spec.node_names)
            # Cross-temporel : nœud temporel ajouté en dernier niveau
            if design.mode == _CROSS_TEMPORAL:
                labels = [(*label, key[1]) for label, key in zip(labels, structure.node_keys)]
                names.append(design.temporal_name)
            # Index simple pour un niveau unique, MultiIndex sinon
            rows = (
                pd.Index([label[0] for label in labels], name=names[0])
                if len(names) == 1
                else pd.MultiIndex.from_tuples(labels, names=names)
            )
        # Colonnes étiquetées par les feuilles (dernières lignes)
        return pd.DataFrame(np.array(structure.summing), index=rows, columns=rows[-structure.n_bottom:])
