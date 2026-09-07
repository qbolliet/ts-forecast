"""Mutualized training set of one variable at one imputation stage.

On a panel, a column may be yearly for one entity,
quarterly for a second and monthly for a third, and the last two carry true
values at the very frequency where the first one must be imputed. Ignoring
them means fitting on three anchors what could be fitted on fifty-one.

The rule is a single sentence:

    The training set of a variable ``v`` at a stage ``f`` is mutualized across
    every entity observing ``v``, each contributing at the frequency it
    observes it at, brought back to the stage scale by a divisor of its own
    block.

The component composes that set and nothing else: it never scales, never
aggregates the target, and calls
:meth:`CovariateMaterializer.materialize` exactly once.
"""
# Importation des modules
# Modules de base
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set,
    Tuple, Union,
)

# Manipulation de données
import numpy as np
import pandas as pd

# Producteur unique des features, et lecture des formes par entité
from .covariate_materializer import CovariateMaterializer, _WAY_RANK
# Voies de matérialisation, définies avec l'étape du plan
from .imputation_plan2 import MaterializationWay
# Primitives d'origine de cellule
from .provenance import CellOrigin, max_origin
# Normalisation des fréquences détectées
from ..utils.frequency.utils import normalize_frequency
# Utilitaires de panel : découpage et normalisation des clés d'entité
from ..panel.utils import iter_entity_blocks, normalize_entity_key


# Clé d'entité, tuple normalisé ("()" pour une série temporelle)
EntityKey = Tuple[Any, ...]
# Fréquence d'étape : scalaire, ou une par entité
StageFrequency = Union[str, Mapping[EntityKey, str]]
# Fréquences détectées : par colonne, éventuellement par entité
DetectedFrequencies = Mapping[str, Union[str, Mapping[EntityKey, str]]]
# Masque de fenêtre d'entraînement, lu à une fréquence par entité
TrainingMask = Callable[[Mapping[EntityKey, str]], pd.Series]
# Callback de journalisation injecté
LogCallback = Callable[[str], None]

# Colonnes du tableau intermédiaire d'une ligne candidate
_ROW_COLUMNS: Tuple[str, ...] = ('value', 'freq', 'origin')

# Nom du niveau de fréquence ajouté à l'index du jeu d'entraînement
FREQUENCY_LEVEL_NAME: str = 'frequency'


# Fonction de décomposition d'un index de jeu d'entraînement
def split_training_index(
    index: pd.Index,
    *,
    has_frequency_level: bool,
) -> Tuple[pd.Index, List[EntityKey]]:
    """Split a training grid into its real index and its entity keys.

    Under ``keep_coincident_cells`` the training grid carries an extra
    frequency level placed on the entity side — ``(entity..., freq, date)`` —
    so the entity of a row is ``key[:-2]`` and no longer ``key[:-1]``. This is
    the single decomposition every consumer goes through, so the two shapes
    are read in exactly one place.

    Args:
        index: Training grid, stamped or not.
        has_frequency_level: Whether the grid carries the frequency level,
            read from :attr:`TrainingSet.has_frequency_level` — never sniffed
            from the level names.

    Returns:
        Tuple ``(real_index, entities)``: the ``(entity..., date)`` grid the
        materializer and the scaler read — the input itself when no level was
        added — and the entity key of each row, in grid order. A time series
        yields the degenerate key ``()`` everywhere.

    Examples:
        >>> dates = pd.to_datetime(['2021-12-31', '2021-12-31'])
        >>> stamped = pd.MultiIndex.from_arrays(
        ...     [['Y', 'Q'], dates], names=['frequency', 'date']
        ... )
        >>> real, entities = split_training_index(
        ...     stamped, has_frequency_level=True
        ... )
        >>> list(real), entities
        ([Timestamp('2021-12-31 00:00:00'), Timestamp('2021-12-31 00:00:00')], [(), ()])
    """
    # Grille non estampillée : aucun niveau de fréquence
    if not has_frequency_level:
        if not isinstance(index, pd.MultiIndex):
            return index, [()] * len(index)
        depth = index.nlevels - 1
        return index, [tuple(key[:depth]) for key in index]

    # Grille estampillée : le niveau de fréquence est l'avant-dernier
    real = index.droplevel(-2)
    depth = index.nlevels - 2
    return real, [tuple(key[:depth]) for key in index]


# Jeu d'entraînement mutualisé d'une variable à une étape
@dataclass(frozen=True)
class TrainingSet:
    """Mutualized training set of one variable at one stage.

    The target is handed over raw: scaling is the caller's job, and there is
    exactly one implementation of it — :class:`StageScaler`. The caller
    applies ``target_divisor(produced_freq=row_frequency)`` to :attr:`y` and
    ``feature_divisors(grid_freq=blocks, ways=...)`` to :attr:`X`.

    Attributes:
        X: Features, indexed on the mutualized grid (the union of the block
            grids).
        y: Raw target, never scaled and never aggregated, indexed like
            :attr:`X`.
        row_frequency: Production frequency of each row: ``f_block(e)`` for an
            observed cell, the frequency read in ``imputed_freq_store`` for a
            cell produced by an earlier stage.
        row_origin: :data:`CellOrigin` of each row of :attr:`y`, same two
            sources as :attr:`row_frequency`.
        blocks: Mapping entity -> ``f_block(e)``, the composition of the
            mutualized set. It does not depend on the stage: only the
            divisors do.
        ways: Materialization way retained for each covariate, to be imposed
            as-is on the prediction grid.
        column_origins: Aggregated origin of each covariate, input of the
            ``covariate_taint`` computation.
        has_frequency_level: Whether the index of :attr:`X`, :attr:`y`,
            :attr:`row_frequency` and :attr:`row_origin` carries the extra
            frequency level of ``keep_coincident_cells`` — ``(entity...,
            freq, date)`` instead of ``(entity..., date)``. The single source
            of truth on the grid shape: consumers pass it to
            :func:`split_training_index` rather than sniffing the index.

    Examples:
        >>> empty = TrainingSet(
        ...     X=pd.DataFrame(), y=pd.Series(dtype=float),
        ...     row_frequency=pd.Series(dtype=object),
        ...     row_origin=pd.Series(dtype=object),
        ...     blocks={}, ways={}, column_origins={},
        ... )
        >>> len(empty)
        0
    """

    X: pd.DataFrame
    y: pd.Series
    row_frequency: pd.Series
    row_origin: pd.Series
    blocks: Mapping[EntityKey, str]
    ways: Mapping[str, MaterializationWay]
    column_origins: Mapping[str, CellOrigin]
    has_frequency_level: bool = False

    # Nombre de lignes du jeu, mesure directe du gain de la mutualisation
    def __len__(self) -> int:
        """Number of training rows.

        Returns:
            The length of :attr:`y`
        """
        return len(self.y)


# Composant de composition du jeu d'entraînement mutualisé
class TrainingSetBuilder:
    """Compose the mutualized training set of one variable at one stage.

    Six rule satisfied by this class:

    - **Scope**: every entity carrying at least one observation of the
      column contributes, whatever its frequency for that column, whether the
      column is imputable there, and whatever its own target frequency. An
      entity never observing it contributes nothing and raises nothing.
    - **Block frequency**: ``f_block(e) = f_var(e, column)``, the
      entity's own frequency for that column, without exception — including
      when it is finer than the stage. Training then runs on a grid
      finer than the prediction grid, which is what maximizes the number of
      rows. The blocks therefore do not depend on the stage.
    - **Rows of a block**: one single rule, whatever the position of
      ``f_var(e, column)`` against the stage — the observed cells of the
      column for that entity, its anchors, taken at ``f_block(e)``. No
      aggregation of the target ever happens, hence no aggregation constraint
      and no ``full_periods_only`` effect: an incomplete period costs no row.
      The rows are then restricted by the ``'training'`` window read at the
      block frequency, then filtered by ``eligible_origins``.
    - **Covariates**: one single call to
      :meth:`CovariateMaterializer.materialize`, on the mutualized grid, with
      ``stage_freq={e: f_block(e)}`` and ``record=False``. An entity whose
      block is finer than the stage sees its covariates materialized by a
      lower-ranked way (a quarterly covariate is interpolated onto its monthly
      grid): that is expected and harmless — such an entity is never imputable
      at that stage, hence never present on a prediction grid, and training
      more degraded than prediction is allowed.
    - **Row frequency**: :attr:`TrainingSet.row_frequency` is
      ``f_block(e)`` for an observed row, and the frequency read in
      ``imputed_freq_store`` for a row of origin ``'interpolated'`` or
      ``'model'`` produced by an earlier stage. The position of the block
      against the stage is read in the divisor only: ``> 1`` for a block
      coarser than the stage, ``1.0`` at the stage, fractional for a finer
      block (``get_conversion_factor('Q', 'M') == 1/3``).
    - **One single fit**: the set returned does not depend on the source
      frequency group — it is a function of (column, stage) alone. Hence no
      group and no target-entity parameter here, and one single fit per
      (stage, variable), shared by the plan steps.

    The assumed bias: mutualizing supposes comparable levels across
    entities — a country ten times bigger pulls the target, and
    ``scale_features`` only corrects the frequency scale, never the entity
    one. The arbitrage is settled in favour of volume, and a user wanting one
    model per entity gets it without any parameter, by fitting one imputer per
    entity.

    Args:
        materializer: The :class:`CovariateMaterializer` of the imputer — the
            single producer of features, and the holder of the three stores.
        training_mask: Callable returning the ``kind='training'`` mask at the
            frequency asked for each entity, i.e. the injected form of
            ``ImputationWindowCalculator.get_mask_at_frequency(...,
            kind='training')``, which already accepts one frequency per
            entity. None — the callable itself, or what one call returns —
            applies no window restriction at all. An entity
            absent from the mask returned is left unrestricted (the calculator
            simply omits the entities without a valid fitted mask); for an
            entity present, a date absent from the mask is outside the window
            and its row is dropped.
        log: Logging callback, called with one-line messages. Inert when None.

    Examples:
        >>> materializer = CovariateMaterializer(covariate_strategy='interpolate')
        >>> builder = TrainingSetBuilder(materializer)
        >>> dates = pd.date_range('2021-01-31', periods=12, freq='ME')
        >>> data = pd.DataFrame(
        ...     {'m1': range(12), 'a1': float('nan')}, index=dates, dtype=float
        ... )
        >>> data.loc['2021-12-31', 'a1'] = 120.0
        >>> training = builder.build(
        ...     column='a1', feature_cols=['m1'], stage_freq='M',
        ...     detected_frequencies={'m1': 'M', 'a1': 'Y'},
        ...     source_data=data, eligible_origins={'observed'},
        ... )
        >>> training.blocks
        {(): 'Y'}
        >>> training.y.tolist()
        [120.0]
    """

    # Initialisation : injection pure, aucun état
    def __init__(
        self,
        materializer: CovariateMaterializer,
        training_mask: Optional[TrainingMask] = None,
        log: Optional[LogCallback] = None,
    ) -> None:
        """Store the injected collaborators.

        Args:
            materializer: The imputer's covariate materializer.
            training_mask: Training-window mask provider, or None.
            log: Logging callback, or None.
        """
        # Stockage des collaborateurs injectés
        self.materializer = materializer
        self.training_mask = training_mask
        self.log = log

    # Méthode auxiliaire de journalisation
    def _log(self, message: str) -> None:
        """Emit one line through the injected callback.

        Args:
            message: Message to log. Dropped when no callback was injected.
        """
        # Journalisation inerte tant qu'aucun callback n'est fourni
        if self.log is not None:
            self.log(message)

    # -------------------------------------------------------------------------
    # Périmètre des contributeurs et fréquence de bloc
    # -------------------------------------------------------------------------
    # Méthode de composition des blocs d'entité
    def blocks_of(
        self,
        column: str,
        detected_frequencies: DetectedFrequencies,
        source_data: pd.DataFrame,
    ) -> Dict[EntityKey, str]:
        """Compute ``f_block(e)`` for every entity observing the column.

        The block frequency is the entity's own
        frequency for the column, read in ``detected_frequencies`` under its
        per-entity form: no ``min`` of frequencies, and no comparison against
        the stage frequency ever takes part in the decision, so the blocks are
        stage-independent.

        Args:
            column: Column whose training set is being composed.
            detected_frequencies: Detected frequency of each column, scalar or
                per entity — a detected frequency is a property of the
                (entity, column) pair, never of the column alone.
            source_data: Input data, time series or panel. Never modified.

        Returns:
            Mapping entity key -> normalized block frequency, in the entity
            order of the data. Entities observing the column at no date at
            all, and entities whose frequency is unknown, are absent.

        Raises:
            KeyError: If ``column`` is absent from ``source_data``.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=2, freq='ME')
            >>> idx = pd.MultiIndex.from_product([['DE', 'FR'], dates])
            >>> df = pd.DataFrame(
            ...     {'v': [1.0, 2.0, float('nan'), float('nan')]}, index=idx
            ... )
            >>> TrainingSetBuilder(CovariateMaterializer()).blocks_of(
            ...     'v', {'v': {('DE',): 'M', ('FR',): 'Y'}}, df)
            {('DE',): 'M'}
        """
        # Vérification de la présence de la colonne cible
        if column not in source_data.columns:
            raise KeyError(f"Column {column!r} missing from source_data")

        # Initialisation du dictionnaire des blocs
        blocks: Dict[EntityKey, str] = {}
        # Parcours des entités du jeu, dans leur ordre d'apparition
        for entity, _mask, block in iter_entity_blocks(source_data):
            # Normalisation de la clé d'entité
            key = normalize_entity_key(entity)
            # Une entité n'observant jamais la colonne ne contribue rien,
            # et ne lève rien
            if block[column].notna().sum() == 0:
                continue
            # Fréquence propre de l'entité pour cette colonne, lue sous sa
            # forme par entité — aucune comparaison à la fréquence d'étape
            freq = CovariateMaterializer._column_frequency(
                detected_frequencies, column, key
            )
            # Fréquence indétectable : l'entité ne peut porter aucun diviseur
            if freq is None:
                self._log(
                    f"[TrainingSetBuilder] {column!r} : entity {key!r} has no "
                    "detected frequency, block dropped"
                )
                continue
            # Ajout de la fréquence normalisée aux blocs
            blocks[key] = normalize_frequency(freq)
        return blocks

    # -------------------------------------------------------------------------
    # Lignes d'un bloc
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de composition des lignes candidates d'une entité
    def _candidate_rows(
        self,
        column: str,
        block: pd.DataFrame,
        f_block: str,
        mirror_block: Optional[pd.DataFrame],
        keep_coincident_cells: bool = False,
    ) -> pd.DataFrame:
        """Compose the candidate rows of one block, before any filtering.

        The observed cells of the column for this
        entity, at ``f_block(e)``. No aggregation, hence no aggregation
        constraint and no ``full_periods_only`` effect. The cells produced by
        an earlier stage are added from the mirror, carrying their own
        production frequency and origin.

        Two cells of one entity may share a date without describing the same
        period — the 31st of December is both a year end and a quarter end.
        ``keep_coincident_cells`` decides between them: False (the historical
        behaviour) keeps the observed one alone, True keeps both, which the
        caller then distinguishes by the frequency level of the grid.

        Args:
            column: Column being composed.
            block: This entity's source data, date-indexed.
            f_block: Block frequency of this entity.
            mirror_block: This entity's mirror block — a frame with columns
                ``value`` / ``freq`` / ``origin`` — or None when the column
                produced nothing for that entity yet.
            keep_coincident_cells: Whether a produced cell falling on a date
                already carrying an observation is kept. The component never
                learns WHY it is asked to: the decision belongs to the caller,
                exactly like ``eligible_origins``.

        Returns:
            Date-indexed frame with columns ``value``, ``freq`` and
            ``origin``, sorted by date. The sort is stable, so an observed
            cell always precedes the finer produced cell it coincides with.
        """
        # Cellules observées : leur fréquence est celle du bloc, leur origine
        # est 'observed' par construction
        observed = block[column].dropna()
        rows = pd.DataFrame(
            {
                'value': observed.astype(float),
                'freq': f_block,
                'origin': 'observed',
            },
            index=observed.index,
            columns=list(_ROW_COLUMNS),
        )

        # Cellules produites par une étape antérieure : fréquence et origine
        # lues dans les registres
        if mirror_block is not None and not mirror_block.empty:
            # Départage des cellules coïncidentes: une période que
            # non recalée n'est pas colinéaire non
            # plus, mais une règle dépendant des valeurs serait irreproductible
            # entre le fit et le transform.
            coincident = mirror_block.index.isin(rows.index)
            extra = mirror_block.loc[
                slice(None) if keep_coincident_cells else ~coincident,
                list(_ROW_COLUMNS),
            ]
            extra = extra[extra['value'].notna()]
            if not extra.empty:
                rows = pd.concat([rows, extra])

        # Tri stable : à date égale, l'observation précède la cellule produite
        return rows.sort_index(kind='stable')

    # Méthode auxiliaire de découpage des lignes en couches homogènes
    @staticmethod
    def _frequency_layers(
        rows_per_entity: Mapping[EntityKey, pd.DataFrame],
        blocks: Mapping[EntityKey, str],
    ) -> List[Dict[EntityKey, str]]:
        """Split the rows into layers of one frequency per entity.

        A block is homogeneous only as long as axis 2 injects nothing into
        it: a cell produced at an earlier, finer stage sits on a row of that
        stage's frequency, not of its block's. A layer is one frequency per
        contributing entity — the shape :meth:`CovariateMaterializer.materialize`
        and the training mask both read — and the block frequency always
        opens the list, so a homogeneous set yields exactly **one** layer,
        equal to ``blocks``: the axis-2-off path keeps its single call.

        Args:
            rows_per_entity: Candidate rows of each entity, carrying a
                ``freq`` column.
            blocks: Block frequency of each entity.

        Returns:
            Ordered list of ``{entity: frequency}`` bindings covering every
            row exactly once.

        Examples:
            >>> builder._frequency_layers(rows, blocks)     # doctest: +SKIP
            [{('FR',): 'Y', ('DE',): 'Q'}, {('FR',): 'Q'}]
        """
        # Fréquences de chaque entité, celle de son bloc en tête
        # Initialisation du dictionnaire résultat
        per_entity: Dict[EntityKey, List[str]] = {}
        # Parcours des entités
        for entity, rows in rows_per_entity.items():
            # Extraction des fréqunces comprimes dans le bloc
            f_block = blocks.get(entity)
            frequencies = [f_block] if f_block is not None else []
            # Ajout des fréquences manquantes
            for frequency in pd.unique(rows['freq'].dropna().to_numpy(dtype=object)):
                if frequency not in frequencies:
                    frequencies.append(frequency)
            per_entity[entity] = frequencies

        # Couches : la i-ème fréquence de chaque entité qui en porte une
        depth = max((len(f) for f in per_entity.values()), default=0)
        return [
            {
                entity: frequencies[rank]
                for entity, frequencies in per_entity.items()
                if rank < len(frequencies)
            }
            for rank in range(depth)
        ]

    # Méthode auxiliaire de sélection des lignes d'une couche
    @staticmethod
    def _rows_of_layer(
        rows: pd.DataFrame,
        entity: EntityKey,
        layer: Mapping[EntityKey, str],
    ) -> np.ndarray:
        """Return the boolean selector of one entity's rows in one layer.

        Args:
            rows: Candidate rows of the entity, carrying a ``freq`` column.
            entity: Entity key of the block.
            layer: Binding of the layer.

        Returns:
            Boolean array, all False when the entity does not reach the
            layer.
        """
        # Entité absente de la couche : aucune ligne
        frequency = layer.get(entity)
        if frequency is None:
            return np.zeros(len(rows), dtype=bool)
        return (rows['freq'] == frequency).to_numpy(dtype=bool)

    # Méthode auxiliaire de restriction par la fenêtre d'entraînement
    def _restrict_to_window(
        self,
        candidates: Mapping[EntityKey, pd.DataFrame],
        blocks: Mapping[EntityKey, str],
    ) -> Dict[EntityKey, pd.DataFrame]:
        """Restrict the candidate rows to the ``'training'`` window.

        The mask is read at the frequency of **each row**, which is the same
        unifying rule as the scale divisors: an observed cell carries its
        block frequency ``f_block(e)``, a cell produced by an earlier stage
        carries its production frequency. Reading the window at the block
        frequency alone would leave every finer produced cell outside the
        mask's own grid — hence silently dropped — and no target could ever
        train on its own earlier imputations.

        One call per layer (:meth:`_frequency_layers`), never one per row: a
        homogeneous set yields a single layer equal to the blocks, and the
        call is then the historical one.

        Args:
            candidates: Candidate rows of each entity.
            blocks: Block frequency of each entity.

        Returns:
            The candidate rows restricted to the window, entity by entity.
            A copy of the input mapping when no ``training_mask`` was
            injected.
        """
        # Aucune fenêtre injectée : aucune restriction
        if self.training_mask is None:
            return dict(candidates)

        # Un appel par couche. Un retour None dit qu'aucune fenêtre n'est
        # calculable — calculateur non ajusté, fréquence de bloc
        # inconvertible : aucune restriction plutôt que le rejet silencieux
        # de toutes les lignes
        layers = self._frequency_layers(candidates, blocks)
        masks: List[Tuple[Dict[EntityKey, str], Optional[Dict[EntityKey, pd.Series]]]] = []
        for layer in layers:
            mask = self.training_mask(dict(layer))
            masks.append((
                layer,
                None if mask is None else {
                    normalize_entity_key(entity): entity_block
                    for entity, _mask, entity_block in iter_entity_blocks(mask)
                },
            ))

        # Restriction bloc par bloc, couche par couche
        restricted: Dict[EntityKey, pd.DataFrame] = {}
        for entity, rows in candidates.items():
            keep = np.zeros(len(rows), dtype=bool)
            for layer, mask_blocks in masks:
                in_layer = self._rows_of_layer(rows, entity, layer)
                if not in_layer.any():
                    continue
                # Masque correspondant aux observations de l'entité. Entité
                # absente du masque : le calculateur omet les entités sans
                # masque ajusté, aucune restriction n'est alors définissable
                entity_mask = None if mask_blocks is None else mask_blocks.get(entity)
                if entity_mask is None:
                    keep |= in_layer
                    continue
                # Date absente du masque : hors fenêtre, la ligne est écartée.
                in_window = entity_mask.reindex(rows.index).eq(True).to_numpy(dtype=bool)
                keep |= in_layer & in_window
            restricted[entity] = rows[keep]
        return restricted

    # -------------------------------------------------------------------------
    # Assemblage de la grille mutualisée
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de construction de l'index de la grille mutualisée
    @staticmethod
    def _mutualized_index(
        rows_per_entity: Mapping[EntityKey, pd.DataFrame],
        source_data: pd.DataFrame,
        keep_coincident_cells: bool = False,
    ) -> pd.Index:
        """Assemble the union of the block grids into one index.

        Under ``keep_coincident_cells`` the grid gains a frequency level
        placed on the entity side — the block becomes the ``(entity,
        frequency)`` pair, which is exactly the key of a layer —
        so two cells of one entity sharing a date but describing different
        periods each get their own key.

        Args:
            rows_per_entity: Retained rows of each entity, date-indexed and
                carrying a ``freq`` column.
            source_data: Input data, whose index shape and names are
                reproduced.
            keep_coincident_cells: Whether the frequency level is stamped in.

        Returns:
            Without the level: a ``DatetimeIndex`` for a time series, a
            ``MultiIndex`` ``(entity..., date)`` for a panel. With it: a
            ``MultiIndex`` ``(freq, date)`` or ``(entity..., freq, date)``,
            the level named ``'frequency'`` and inserted before the date
            level. Empty and of the source shape when no entity contributes.
        """
        # Aucun contributeur : grille vide, de la forme de la source
        if not rows_per_entity:
            return source_data.index[:0]

        # Noms de niveaux de la source, la date en dernier
        source_names = (
            list(source_data.index.names)
            if isinstance(source_data.index, pd.MultiIndex)
            else [source_data.index.name]
        )

        # Grille estampillée : la fréquence de chaque ligne devient un niveau,
        # inséré avant la date
        if keep_coincident_cells:
            stamped: List[Tuple[Any, ...]] = [
                (*entity, frequency, date)
                for entity, rows in rows_per_entity.items()
                for date, frequency in zip(rows.index, rows['freq'])
            ]
            names = source_names[:-1] + [FREQUENCY_LEVEL_NAME, source_names[-1]]
            return pd.MultiIndex.from_tuples(stamped, names=names)

        # Série temporelle : entité dégénérée unique, index de dates
        if list(rows_per_entity) == [()]:
            dates = pd.DatetimeIndex(rows_per_entity[()].index)
            return dates.rename(source_data.index.name)

        # Panel : union des grilles de bloc, entités dans l'ordre des données
        tuples: List[Tuple[Any, ...]] = [
            (*entity, date)
            for entity, rows in rows_per_entity.items()
            for date in rows.index
        ]
        return pd.MultiIndex.from_tuples(tuples, names=source_names)

    # -------------------------------------------------------------------------
    # Méthode publique de composition du jeu
    # -------------------------------------------------------------------------
    # Méthode de composition du jeu d'entraînement mutualisé
    def build(
        self,
        *,
        column: str,
        feature_cols: Sequence[str],
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        source_data: pd.DataFrame,
        eligible_origins: Iterable[CellOrigin],
        materialization: Optional[Mapping[str, MaterializationWay]] = None,
        keep_coincident_cells: bool = False,
    ) -> TrainingSet:
        """Compose the mutualized training set of one variable at one stage.

        The six rules of the class: blocks, rows,
        window and origin filter, then one call to the materializer. The
        target comes back raw, with its per-row production frequency, and
        the set depends on no frequency group.

        Args:
            column: Column being imputed, the target of the set.
            feature_cols: Covariate columns to materialize, in output order.
            stage_freq: Frequency of the stage, scalar or per entity. It
                governs neither the blocks nor the rows — only the divisors
                the caller then applies depend on it. It is read for the
                log line alone.
            detected_frequencies: Detected frequency of each column, scalar or
                per entity.
            source_data: Input data, time series or panel. Never modified.
            eligible_origins: Cell origins admissible in ``y`` — the
                ``ELIGIBLE_ORIGINS`` entry of the stage. The
                origin filter and the mutualization are orthogonal: this
                decides wihich cells of an entity are eligible. The other dimension decides
                which entities contribute and at which frequency.
            materialization: Ways to impose on the covariates, one entry per
                column of ``feature_cols``. None (default) lets the
                materializer choose them on the block grid. The caller of an
                imputation stage passes the ways decided on the prediction
                grid: a covariate served by ``covariate_fallback``
                at predict must be prepared the same way at fit, even when its
                anchors would suffice. :meth:`CovariateMaterializer._applicable_way`
                then brings each imposed way back to what the entity's own
                block frequency allows — an ``'interpolate'`` way on a block
                grid no finer than the covariate becomes the rank-1 identity
                or aggregation.
            keep_coincident_cells: Whether two cells of one entity sharing a
                date but describing different periods — a yearly anchor and
                the fourth quarterly imputation of that year — are BOTH kept.
                False (default) keeps the observed one alone and the index
                stays ``(entity..., date)``; True keeps both and the index
                gains a ``'frequency'`` level on the entity side — but only
                where a coincidence actually survives the origin filter, so
                a set holding anchors alone keeps the historical shape and
                the parameter is inert on it. Like
                ``eligible_origins``, the reason belongs to the caller: this
                class never reads ``aggregation_constraint``.

        Returns:
            The :class:`TrainingSet`. Empty — with empty ``blocks`` and
            ``ways`` — when no entity observes the column, without raising.

        Raises:
            KeyError: If ``column`` is absent from ``source_data``.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=12, freq='ME')
            >>> data = pd.DataFrame(
            ...     {'m1': range(12), 'a1': float('nan')}, index=dates, dtype=float
            ... )
            >>> data.loc[['2021-06-30', '2021-12-31'], 'a1'] = [50.0, 120.0]
            >>> builder = TrainingSetBuilder(CovariateMaterializer())
            >>> training = builder.build(
            ...     column='a1', feature_cols=['m1'], stage_freq='M',
            ...     detected_frequencies={'m1': 'M', 'a1': 'Q'},
            ...     source_data=data, eligible_origins={'observed'},
            ... )
            >>> len(training), training.ways
            (2, {'m1': 'aggregate'})
        """
        # Normalisation des paramètres d'appel
        columns = tuple(feature_cols)
        admissible: Set[CellOrigin] = set(eligible_origins)

        # Périmètre des contributeurs et fréquence de bloc
        blocks = self.blocks_of(column, detected_frequencies, source_data)

        # Blocs d'entité de la source et du miroir, lus une seule fois
        source_blocks = {
            normalize_entity_key(entity): block
            for entity, _mask, block in iter_entity_blocks(source_data)
        }
        mirror_blocks = self.materializer._mirror_blocks(column)

        # Lignes candidates de chaque bloc, une seule règle
        candidates = {
            entity: self._candidate_rows(
                column,
                source_blocks[entity],
                f_block,
                mirror_blocks.get(entity),
                keep_coincident_cells=keep_coincident_cells,
            )
            for entity, f_block in blocks.items()
        }

        # Restriction par la fenêtre 'training', lue à la fréquence du bloc
        candidates = self._restrict_to_window(candidates, blocks)

        # Filtre d'origine, appliqué à l'identique dans chaque bloc,
        # puis abandon des blocs devenus vides
        rows_per_entity = {
            entity: rows[rows['origin'].isin(admissible).to_numpy()]
            for entity, rows in candidates.items()
        }
        rows_per_entity = {
            entity: rows for entity, rows in rows_per_entity.items() if not rows.empty
        }

        # Estampille du niveau de fréquence : elle n'a lieu que lorsqu'elle
        # sert, c'est-à-dire lorsque deux cellules retenues d'une même entité
        # partagent une date. Sans coïncidence — le cas de tout jeu sans axe 2,
        # où y_train ne tient que des ancres — l'index reste inchangé
        stamp = keep_coincident_cells and any(
            bool(rows.index.duplicated().any()) for rows in rows_per_entity.values()
        )

        # Grille mutualisée : union des grilles de bloc. Sous l'estampille, deux
        # grilles coexistent — la réelle, seule remise au matérialiseur, et
        # l'estampillée, portée par le jeu rendu
        grid = self._mutualized_index(rows_per_entity, source_data, stamp)
        real_grid = (
            self._mutualized_index(rows_per_entity, source_data) if stamp else grid
        )

        # Cible brutes et métadonnées de ligne, dans l'ordre de la grille
        stacked = (
            pd.concat(list(rows_per_entity.values()))
            if rows_per_entity
            else pd.DataFrame(columns=list(_ROW_COLUMNS))
        )
        y = pd.Series(stacked['value'].to_numpy(dtype=float), index=grid, name=column)
        row_frequency = pd.Series(
            stacked['freq'].to_numpy(dtype=object), index=grid, name='row_frequency'
        )
        row_origin = pd.Series(
            stacked['origin'].to_numpy(dtype=object), index=grid, name='row_origin'
        )

        # Journalisation de la composition du jeu, étape à l'appui
        self._log(
            f"[TrainingSetBuilder] {column!r} at stage {stage_freq!r} : "
            f"{len(y)} rows from blocks {blocks}"
        )

        # Aucun contributeur : jeu vide, sans matérialisation ni levée
        if not rows_per_entity:
            return TrainingSet(
                X=pd.DataFrame(index=grid, columns=list(columns), dtype=float),
                y=y,
                row_frequency=row_frequency,
                row_origin=row_origin,
                blocks=blocks,
                ways={},
                column_origins={},
                has_frequency_level=stamp,
            )

        # Un appel au matérialiseur par couche, sur la sous-grille de la
        # couche, à la fréquence de chaque entité qui l'atteint. Une seule
        # couche — le cas dès que l'axe 2 n'injecte rien — rend l'appel unique
        # sur la grille mutualisée et aux fréquences de bloc.
        # Extraction des couches
        layers = self._frequency_layers(rows_per_entity, blocks)
        # Initialisation des étapes
        frames: List[pd.DataFrame] = []
        # Initialisation des matérialisations
        ways: Dict[str, MaterializationWay] = {}
        # Initialisation des origines
        column_origins: Dict[str, CellOrigin] = {}
        # Parcours des couches
        for layer in layers:
            # Masque des lignes dans la couche
            in_layer = np.concatenate([
                self._rows_of_layer(rows, entity, layer)
                for entity, rows in rows_per_entity.items()
            ])
            if not in_layer.any():
                continue
            # Matérialisation
            frame, sub_ways, sub_origins = self.materializer.materialize(
                columns=columns,
                grid_index=real_grid[in_layer],
                stage_freq=dict(layer),
                detected_frequencies=detected_frequencies,
                source_data=source_data,
                materialization=materialization,
                record=False,
            )
            # Estampille de la fréquence APRÈS la matérialisation, à
            # l'empilement : une couche ne porte qu'une fréquence par entité,
            # c'est donc un ajout de niveau constant, jamais une jointure
            if stamp:
                frame = frame.set_axis(grid[in_layer], axis=0)
            frames.append(frame)
            # Réduction sur les couches : la voie la plus dégradée et
            # l'origine la plus souillée, mêmes règles que la réduction par
            # entité de "decide_ways"
            for name, way in sub_ways.items():
                previous = ways.get(name)
                if previous is None or _WAY_RANK[way] > _WAY_RANK[previous]:
                    ways[name] = way
            for name, origin in sub_origins.items():
                column_origins[name] = max_origin(
                    [column_origins.get(name, 'observed'), origin]
                )
        X = (
            pd.concat(frames).reindex(grid)
            if frames
            else pd.DataFrame(index=grid, columns=list(columns), dtype=float)
        )

        # Voies rendues : celles imposées quand il y en a, pour que l'appelant
        # gèle dans l'étape la voie de la grille de prédiction et non son
        # image dégradée sur les grilles de bloc
        if materialization is not None:
            ways = {column: materialization[column] for column in columns}

        return TrainingSet(
            X=X,
            y=y,
            row_frequency=row_frequency,
            row_origin=row_origin,
            blocks=blocks,
            ways=ways,
            column_origins=column_origins,
            has_frequency_level=stamp,
        )

    # Représentation lisible du composant
    def __repr__(self) -> str:
        """Return a readable representation of the wiring.

        Returns:
            One-line representation naming the materializer and whether a
            training window was injected.
        """
        return (
            f"TrainingSetBuilder(materializer={self.materializer!r}, "
            f"training_mask={'set' if self.training_mask is not None else None})"
        )
