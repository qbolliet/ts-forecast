"""Sklearn adapter for hierarchicalforecast reconciliation: ``HierarchicalForecastAdapter``.

``X`` holds the base forecasts of every node of a hierarchy (one column per base
model) and ``y`` the realised values, in a long layout: one row per node and date,
the node carried by index levels placed before the date. ``fit`` builds the summing
matrix of each copy of the hierarchy and keeps the training rows that reconcilers
learning from in-sample errors need; ``predict`` reconciles the base forecasts of
the rows of ``X`` date by date and returns them in the row order of ``X``;
``reconcile`` and ``diagnose`` expose every base model and reconciler.
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


# Fonction d'import différé de hierarchicalforecast
def _import_hierarchicalforecast() -> Tuple[ModuleType, ModuleType, ModuleType]:
    """Import the core, methods and utils modules of hierarchicalforecast."""
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
        structure: Summing matrix, levels and nodes of the copy.
        insample: Training rows in the format of hierarchicalforecast
            (``unique_id``, ``ds``, ``y`` and one column per base model).
        missing_nodes: Nodes without any training row with a target.
        balanced: Whether every node has a row at every training date.
    """

    # Matrice de sommation, niveaux et nœuds de la copie
    structure: Structure
    # Lignes d'entraînement au format de hierarchicalforecast
    insample: pd.DataFrame
    # Nœuds sans aucune ligne d'entraînement à cible renseignée
    missing_nodes: Tuple[Hashable, ...]
    # Présence d'une ligne pour chaque nœud à chaque date d'entraînement
    balanced: bool


# Classe de localisation des lignes dans les copies de la hiérarchie
@dataclass(frozen=True, eq=False)
class _RowNodes:
    """Copy and node of each row of a dataset.

    Attributes:
        copy_keys: Distinct copy keys, in order of first appearance.
        structures: Structure of each copy (same order).
        row_copy: Position in ``copy_keys`` of the copy of each row.
        row_node: Position of the node of each row in the structure of its copy.
        row_level: Position of the level of the node of each row.
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


# Fonction de séparation des combinaisons de niveaux d'index en copies et nœuds
def _split_entities(layout: TemporalLayout, spec: HierarchySpec) -> Tuple[List[tuple], np.ndarray, List[Hashable]]:
    """Return the distinct copy keys, the copy of each entity and the node key of each entity."""
    # Dictionnaire clé de copie -> code (ordre de première apparition)
    lookup: Dict[tuple, int] = {}
    # Code de copie et clé de nœud de chaque combinaison distincte de niveaux d'index
    entity_copy, node_keys = [], []
    # Une seule analyse par combinaison distincte, diffusée ensuite par les codes des lignes
    for entity in layout.entities:
        # Séparation des étiquettes de copie et de nœud
        copy_key, node_key = spec.split_key(entity)
        # Attribution d'un nouveau code à une copie jamais rencontrée
        entity_copy.append(lookup.setdefault(copy_key, len(lookup)))
        node_keys.append(node_key)
    # Clés de copie dans l'ordre de leurs codes
    return list(lookup), np.array(entity_copy, dtype=np.intp), node_keys


# Fonction de résolution des colonnes de prévisions de base et d'intervalle
def _resolve_columns(X: pd.DataFrame, forecast_columns: Any) -> Tuple[List[Hashable], Dict[tuple, Hashable]]:
    """Return the base forecast columns and the interval columns ``(model, side, level) -> column``."""
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
    """Return None, ``'mean'`` or the base column position of each level of the hierarchy."""
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
        temporal_hierarchy: Temporal hierarchy ``{name: number of steps}``;
            temporal reconciliation is not implemented and requires None.

    Attributes:
        reconciler_: Copy of the reconciler used by ``predict``.
        hierarchy_levels_: Names of the levels of the hierarchy, most
            aggregated first (keys of a ``model_combination`` mapping).
        copies_: Keys of the copies of the hierarchy seen at fit (``((),)``
            without copy levels).
        forecast_columns_: Base forecast columns of ``X``.
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
    ):
        # Stockage des paramètres tels quels (convention sklearn : validation au fit)
        self.hierarchy = hierarchy
        self.reconciler = reconciler
        self.node_level = node_level
        self.aggregate_label = aggregate_label
        self.model_combination = model_combination
        self.forecast_columns = forecast_columns
        self.temporal_hierarchy = temporal_hierarchy

    # ------------------------------------------------------------------
    # Méthodes auxiliaires
    # ------------------------------------------------------------------

    # Vérification d'un réconciliateur
    @staticmethod
    def _check_reconciler(reconciler: Any, structures: Sequence[Structure], level_names: Sequence[str]) -> None:
        """Refuse an object that is no reconciler, or a reconciler the structure does not allow."""
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
        """Return the reconcilers of a call, checked against the fitted structures."""
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
            self._check_reconciler(reconciler, structures, self.hierarchy_levels_)
        return reconcilers

    # Localisation des lignes d'un jeu de données à prédire
    def _locate_rows(self, layout: TemporalLayout, needs_insample: bool) -> _RowNodes:
        """Locate the copy and the node of each row of ``X``."""
        # Copies et nœuds des combinaisons de niveaux d'index de X
        copy_keys, entity_copy, node_keys = _split_entities(layout, self._spec)
        # Structure de chaque copie de X (None pour une copie inexploitable)
        structures = []
        # Position du nœud, puis de son niveau, pour chaque combinaison de niveaux d'index
        entity_node = np.empty(len(node_keys), dtype=np.intp)
        entity_level = np.empty(len(node_keys), dtype=np.intp)
        # Copies de X inexploitables
        unknown_copies = []
        # Parcours des copies de X : choix de la structure, puis localisation des nœuds
        for code, copy_key in enumerate(copy_keys):
            # Données ajustées de la copie, None si elle est inconnue
            record = self._copies.get(copy_key)
            if record is not None:
                # Cas d'une copie connue
                structure = record.structure
            elif self._spec.kind == "tree" and not needs_insample:
                # Copie inconnue d'un arbre : structure connue sans données d'entraînement
                structure = self._spec.tree
            else:
                # Cas d'une copie inexploitable : signalée à la fin du parcours
                unknown_copies.append(copy_key)
                structures.append(None)
                continue
            # Combinaisons de niveaux d'index appartenant à la copie
            members = np.flatnonzero(entity_copy == code)
            # Position de leurs nœuds dans la structure
            positions = structure.locate([node_keys[member] for member in members])
            # Refus des nœuds absents de la structure
            if (positions < 0).any():
                unknown = [node_keys[member] for member in members[positions < 0]]
                where = f" of the copy {copy_key!r}" if copy_key else ""
                raise ValueError(f"Nodes{where} unknown at fit: {_preview(unknown)}")
            # Enregistrement du nœud et de son niveau
            entity_node[members] = positions
            entity_level[members] = structure.node_levels[positions]
            structures.append(structure)
        # Refus des copies inexploitables, avec la raison propre à la forme de la hiérarchie
        if unknown_copies:
            reason = (
                "the reconciliation needs their training rows"
                if self._spec.kind == "tree"
                else "their nodes are learnt from the training rows"
            )
            raise ValueError(f"Copies of the hierarchy unknown at fit ({reason}): {_preview(unknown_copies)}")
        # Diffusion à chaque ligne par le code de sa combinaison d'étiquettes
        return _RowNodes(
            copy_keys=tuple(copy_keys),
            structures=tuple(structures),
            row_copy=entity_copy[layout.codes],
            row_node=entity_node[layout.codes],
            row_level=entity_level[layout.codes],
        )

    # Prévisions de base des modèles d'un appel
    @staticmethod
    def _models(
        X: pd.DataFrame, columns: Sequence[Hashable], combination: Any, row_level: np.ndarray, which: str
    ) -> List[Tuple[Hashable, str, np.ndarray]]:
        """Return ``(name, internal name, values)`` of the base models of a call.

        ``which='target'``: the forecast that ``predict`` reconciles (the
        combined forecast, or the single base column); ``'all'``: every base
        column, then the combined forecast.
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
        """Levels for which ``X`` holds both interval columns of a base column."""
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
        """Interval columns of the base models, named after their internal names."""
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
        """Reconcile the rows of ``X``; return ``X``, the values, their names and the diagnostics per copy."""
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
        # Copie, nœud et niveau de chaque ligne
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
            # Lignes de X de la copie, avec leurs nœuds et leurs dates
            members = np.flatnonzero(rows.row_copy == code)
            nodes = rows.row_node[members]
            dates = layout.dates[members]
            # Vérification de la présence de tous les nœuds à chaque date
            self._check_complete(structure, nodes, dates, copy_key)
            # Construction des prévisions de base au format de hierarchicalforecast : identifiant, date,
            # modèles, puis intervalles
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
            # Sortie triée par nœud puis date : réalignement par clés sur les lignes de X
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
        """Replace the internal model name that prefixes a column with the name of the base model."""
        # Découpage du nom '<modèle interne>/<réconciliateur...>'
        internal, separator, rest = column.partition("/")
        # Nom inchangé pour un modèle interne inconnu
        return f"{names.get(internal, internal)}{separator}{rest}"

    # Contrôle de la présence de tous les nœuds à chaque date
    @staticmethod
    def _check_complete(structure: Structure, nodes: np.ndarray, dates: pd.DatetimeIndex, copy_key: tuple) -> None:
        """Refuse a date that lacks the base forecast of a node of the copy."""
        # Code de la date de chaque ligne et calendrier des dates distinctes
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
                ``(copy..., node..., date)``; rows in any order.
            y: Realised values of the same rows (every node, aggregates
                included).

        Returns:
            The fitted adapter.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            NotImplementedError: If ``temporal_hierarchy`` is set.
            TypeError: If ``X`` is not a ``DataFrame``, if its dates are not
                ``datetime64``, or if ``reconciler`` is not a hierarchicalforecast
                reconciler.
            ValueError: If ``hierarchy`` is missing or malformed, if a row
                matches no node of the hierarchy, if ``y`` is not a single
                numeric target aligned with ``X``, if the base forecast columns
                or ``model_combination`` are invalid, or if the reconciler does
                not suit the hierarchy or lacks training rows of a node.
        """
        # Import différé des méthodes de réconciliation
        _, methods, _ = _import_hierarchicalforecast()
        # Vérification des paramètres indispensables
        if self.temporal_hierarchy is not None:
            raise NotImplementedError(
                "Temporal reconciliation is not implemented: temporal_hierarchy must be None"
            )
        if self.hierarchy is None:
            raise ValueError("hierarchy is required: a list of lists of index levels or a {parent: [children]} mapping")
        # Vérification des entrées et analyse de l'index (aucune grille exigée)
        X, y, layout = check_adapter_input(self, X, y, reset=True, detect_freq=False)
        # Vérification d'une cible unique
        if y is None or y.ndim != 1:
            raise ValueError("y must be a single target: a Series indexed like X")
        # Valeurs de la cible, alignées sur les lignes de X
        target = np.asarray(y, dtype=np.float64)

        # Analyse de la hiérarchie à partir des noms des niveaux d'index
        spec = parse_hierarchy(self.hierarchy, list(X.index.names), self.node_level, self.aggregate_label)
        # Colonnes des prévisions de base et de leurs intervalles
        base, intervals = _resolve_columns(X, self.forecast_columns)
        # Mode de combinaison des modèles de base
        combination = _resolve_combination(self.model_combination, spec.level_names, base)
        # Copie du réconciliateur (BottomUp par défaut) : l'objet reçu n'est jamais modifié
        reconciler = methods.BottomUp() if self.reconciler is None else copy.deepcopy(self.reconciler)

        # Structure de chaque copie, depuis les nœuds de ses lignes
        copy_keys, entity_copy, node_keys = _split_entities(layout, spec)
        # Structure de chaque copie, dans l'ordre de leurs codes
        structures = []
        # Position du nœud de chaque combinaison de niveaux d'index
        entity_node = np.empty(len(node_keys), dtype=np.intp)
        # Parcours des copies : construction de leur structure depuis leurs nœuds observés
        for code in range(len(copy_keys)):
            # Combinaisons de niveaux d'index de la copie et leurs clés de nœud
            members = np.flatnonzero(entity_copy == code)
            keys = [node_keys[member] for member in members]
            structure = spec.build_structure(keys)
            # Position de chaque nœud dans la structure
            entity_node[members] = structure.locate(keys)
            structures.append(structure)

        # Vérification de la compatibilité du réconciliateur avec les structures
        self._check_reconciler(reconciler, structures, spec.level_names)

        # Lignes d'entraînement de chaque copie au format de hierarchicalforecast
        row_copy, row_node = entity_copy[layout.codes], entity_node[layout.codes]
        # Niveau du nœud de chaque ligne
        row_level = np.empty(len(X), dtype=np.intp)
        # Parcours des copies : niveau des nœuds de leurs lignes
        for code, structure in enumerate(structures):
            members = row_copy == code
            row_level[members] = structure.node_levels[row_node[members]]
        # Toutes les prévisions de base, combinaison comprise
        models = self._models(X, base, combination, row_level, "all")
        # Dictionnaire clé de copie -> données ajustées de la copie
        copies = {}
        # Parcours des copies : extraction de leurs lignes d'entraînement
        for code, (copy_key, structure) in enumerate(zip(copy_keys, structures)):
            # Lignes de la copie dont la cible est renseignée
            members = np.flatnonzero((row_copy == code) & np.isfinite(target))
            # Ordre canonique (nœud, date) : l'ajustement ne dépend pas de l'ordre des lignes
            members = members[np.lexsort((layout.dates[members].to_numpy(), row_node[members]))]
            # Lignes d'entraînement au format de hierarchicalforecast
            frame = {
                _ID_COL: structure.ids[row_node[members]],
                _TIME_COL: layout.dates[members],
                _TARGET_COL: target[members],
            }
            frame.update({internal: values[members] for _, internal, values in models})
            # Indicatrice des nœuds disposant d'au moins une ligne
            covered = np.zeros(structure.n_nodes, dtype=bool)
            covered[row_node[members]] = True
            # Nombre de dates d'entraînement de la copie
            n_dates = len(layout.dates[members].unique())
            # Enregistrement des données ajustées (grille équilibrée : chaque nœud présent à chaque date)
            copies[copy_key] = _CopyFit(
                structure=structure,
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
        self._spec = spec
        self._intervals = intervals
        self._combination = combination
        self._copies = copies
        self.hierarchy_levels_ = spec.level_names
        self.forecast_columns_ = base
        self.copies_ = tuple(copy_keys)
        self.reconciler_ = reconciler
        return self

    # Méthode de prédiction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Reconcile the base forecasts of the rows of ``X`` with ``reconciler``.

        Args:
            X: Base forecasts with the columns seen at fit; on each date of each
                copy, one row per node of the hierarchy; rows in any order.

        Returns:
            Reconciled forecasts (mean), shape ``(n_samples,)``, in the row
            order of ``X``.

        Raises:
            ImportError: If hierarchicalforecast is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: If the columns differ from those seen at fit, if a node
                or a copy is unknown, if a date lacks a node, or if a base
                forecast is missing.
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
                ``'permbu'`` or ``'conformal'`` (read the training rows).
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
        copy_names = [
            f"level_{position}" if name is None else name
            for position, name in zip(self._spec.copy_levels, self._spec.copy_names)
        ]
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
            hierarchy levels for a list of levels).

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
        # Choix de la structure : copie ajustée, ou arbre commun pour une copie inconnue
        record = self._copies.get(key)
        if record is not None:
            structure = record.structure
        elif self._spec.kind == "tree":
            structure = self._spec.tree
        else:
            raise ValueError(f"Copy {copy!r} unknown at fit; copies: {_preview(self.copies_)}")
        # Étiquettes des nœuds : index simple pour un arbre, MultiIndex pour des niveaux d'index
        if self._spec.kind == "tree":
            rows = pd.Index(list(structure.node_keys), name=self._spec.node_names[0])
        else:
            rows = pd.MultiIndex.from_tuples(list(structure.node_keys), names=list(self._spec.node_names))
        # Colonnes étiquetées par les feuilles (dernières lignes)
        return pd.DataFrame(np.array(structure.summing), index=rows, columns=rows[-structure.n_bottom:])
