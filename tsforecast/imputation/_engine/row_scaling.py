"""Per-row scaling of the mutualized training sets of ``HighFrequencyImputer``.

``TrainingRows`` decomposes a training grid once (real index, entity of each row, layers of
rows sharing one frequency per entity); ``RowScaler`` wraps the ``StageScaler`` and computes
the per-row divisors of the target and of the covariates, one scaler call per layer on the
real, duplicate-free ``(entity..., date)`` sub-grid, written back by position. ``row_layers``
is the layer enumeration both read.
"""
# Importation des modules
# Modules de base
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Utilitaires du package
from ...panel.utils import factorize_row_keys
from .._components.imputation_plan import MaterializationWay
from .._components.stage_scaler import StageScaler
from .._components.training_set_builder import TrainingSet, split_training_index
from .frequency_layout import ColumnFrequencies, EntityKey, FrequencyLayout, StageFrequency

# Couche de lignes : une fréquence par entité contributrice
Layer = Dict[EntityKey, str]
# Couche et sélecteur booléen de ses lignes sur la grille
LayerRows = Tuple[Layer, np.ndarray]


# Fonction d'énumération des couches d'une grille d'entraînement
def row_layers(
    entity_codes: np.ndarray,
    entities: Sequence[EntityKey],
    row_frequency: pd.Series,
    blocks: Mapping[EntityKey, str],
) -> List[LayerRows]:
    """Enumerate the layers of a training grid, one frequency per entity.

    Same rule as :meth:`TrainingSetBuilder._frequency_layers`, read here on
    the assembled grid rather than on the per-entity frames: for each entity,
    its distinct row frequencies, its block frequency first; the layer of
    rank *k* takes the *k*-th frequency of every entity that has one. A grid
    where axis 2 injected nothing therefore yields exactly one layer. The
    rows are read through integer codes of their entity and frequency: one
    pass per distinct (entity, frequency) pair, never per row.

    Args:
        entity_codes: Entity code of each row, in grid order.
        entities: Entity keys the codes point to.
        row_frequency: Production frequency of each row, in grid order.
        blocks: Block frequency of each contributing entity.

    Returns:
        Ordered list of ``(layer, selector)`` pairs, the selector being a
        boolean array over the grid. Empty layers are dropped; a row whose
        frequency is missing belongs to no layer.

    Examples:
        >>> layers = row_layers(
        ...     np.array([0, 0, 1]), [('FR',), ('DE',)],
        ...     pd.Series(['Y', 'Q', 'Q'], dtype='str'), {('FR',): 'Y', ('DE',): 'Q'},
        ... )
        >>> [(layer, selector.tolist()) for layer, selector in layers]
        [({('FR',): 'Y', ('DE',): 'Q'}, [True, False, True]), ({('FR',): 'Q'}, [False, True, False])]
    """
    # Codes des fréquences de ligne ; une fréquence manquante n'appartient
    # à aucune couche
    freq_codes, freq_labels = pd.factorize(row_frequency, use_na_sentinel=True)
    label_codes = {label: code for code, label in enumerate(freq_labels)}

    # Fréquences de chaque entité, celle de son bloc en tête, puis celles
    # de ses lignes dans l'ordre de première apparition des couples
    per_entity: Dict[int, List[str]] = {}
    _pair_codes, first_rows = factorize_row_keys(entity_codes, freq_codes)
    for row in first_rows:
        entity_code = int(entity_codes[row])
        entity = entities[entity_code]
        ordered = per_entity.setdefault(
            entity_code,
            [blocks[entity]] if entity in blocks else [],
        )
        if freq_codes[row] >= 0 and freq_labels[freq_codes[row]] not in ordered:
            ordered.append(freq_labels[freq_codes[row]])

    # Couches : la i-ème fréquence de chaque entité qui en porte une ; une
    # ligne en fait partie si sa fréquence est celle de son entité au rang i
    layers: List[LayerRows] = []
    depth = max((len(f) for f in per_entity.values()), default=0)
    for rank in range(depth):
        layer = {
            entities[entity_code]: ordered[rank]
            for entity_code, ordered in per_entity.items()
            if rank < len(ordered)
        }
        # Code de fréquence attendu par entité (-2 : aucun, jamais égal)
        expected = np.full(len(entities), -2, dtype=np.intp)
        for entity_code, ordered in per_entity.items():
            if rank < len(ordered):
                expected[entity_code] = label_codes.get(ordered[rank], -2)
        in_layer = (freq_codes >= 0) & (freq_codes == expected[entity_codes])
        if in_layer.any():
            layers.append((layer, in_layer))
    return layers


# Lignes d'un jeu d'entraînement, décomposées une fois
@dataclass(frozen=True)
class TrainingRows:
    """Rows of a mutualized training set, decomposed once for every per-row reader.

    The training grid may carry a frequency level (two coincident cells of
    one entity sharing a date); the materializer and the scaler only ever
    read the real ``(entity..., date)`` grid. Both shapes are decomposed here
    once, by :func:`split_training_index`, and the layers are enumerated once
    for the target, the covariates and the covariate selection alike.

    Attributes:
        row_frequency: Production frequency of each row, indexed on the
            training grid — stamped with a frequency level or not.
        real_index: The ``(entity..., date)`` grid of the same rows, in the
            same order (a ``DatetimeIndex`` on a time series).
        entity_codes: Entity code of each row.
        entities: Entity keys the codes point to.
        blocks: Block frequency of each contributing entity.
        layers: Layers of the grid, as :func:`row_layers` enumerates them.

    Examples:
        >>> dates = pd.to_datetime(['2021-12-31', '2021-12-31', '2022-12-31'])
        >>> stamped = pd.MultiIndex.from_arrays(
        ...     [pd.Index(['Y', 'Q', 'Y'], dtype='str'), dates], names=['frequency', 'date']
        ... )
        >>> training = TrainingSet(
        ...     X=pd.DataFrame(index=stamped), y=pd.Series([120.0, 30.0, 132.0], index=stamped),
        ...     row_frequency=pd.Series(['Y', 'Q', 'Y'], index=stamped, dtype='str'),
        ...     row_origin=pd.Series(index=stamped, dtype=float),
        ...     blocks={(): 'Y'}, ways={}, column_origins={}, has_frequency_level=True,
        ... )
        >>> rows = TrainingRows.of(training)
        >>> rows.real_index.has_duplicates
        True
        >>> [(layer, selector.tolist()) for layer, selector in rows.layers]
        [({(): 'Y'}, [True, False, True]), ({(): 'Q'}, [False, True, False])]
    """

    row_frequency: pd.Series
    real_index: pd.Index
    entity_codes: np.ndarray
    entities: List[EntityKey]
    blocks: Dict[EntityKey, str]
    layers: List[LayerRows]

    # Décomposition d'un jeu d'entraînement
    @classmethod
    def of(cls, training: TrainingSet) -> 'TrainingRows':
        """Decompose the grid of a training set.

        Args:
            training: Training set, its grid shape read from
                :attr:`TrainingSet.has_frequency_level`, never sniffed from
                the level names.

        Returns:
            The decomposed rows.
        """
        # Grille réelle et entité de chaque ligne : le niveau de fréquence
        # n'est jamais lu par le matérialiseur ni par le scaler
        real_index, entity_codes, entities = split_training_index(
            training.X.index, has_frequency_level=training.has_frequency_level
        )
        blocks = dict(training.blocks)
        return cls(
            row_frequency=training.row_frequency,
            real_index=real_index,
            entity_codes=entity_codes,
            entities=entities,
            blocks=blocks,
            layers=row_layers(entity_codes, entities, training.row_frequency, blocks),
        )


# Mise à l'échelle par ligne d'un jeu d'entraînement mutualisé
class RowScaler:
    """Per-row divisors of a mutualized training set, layer by layer.

    The grid of a mutualized training set is homogeneous only as long as
    axis 2 injects nothing into it: a cell produced at an earlier, finer
    stage sits on a row of that stage's frequency, not of its block's. Every
    divisor is therefore read at the frequency of its row, one layer at a
    time: one :class:`StageScaler` call per layer, never one per row, so a
    grid where axis 2 injected nothing is a single layer, hence a single call
    at the block binding.

    The scaler only ever sees the real, duplicate-free ``(entity..., date)``
    sub-grid of a layer: under a stamped grid two rows share a date, and the
    scaler — which writes its results back by label — would set both. The
    results are written back by position on the training grid.

    Args:
        stage_scaler: Stage scaler of the pass, sole implementation of the
            divisor arithmetic.
        layout: Frequency layout of the fit, which tells whether the grid
            carries entities.

    Examples:
        >>> from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
        >>> layout = FrequencyLayout(
        ...     is_panel=False, detected_frequencies={'a1': 'Y'}, undetected=(),
        ...     misaligned=(), target_frequency='M', categories={}, progression=['M'],
        ... )
        >>> RowScaler(StageScaler(), layout).layer_binding({(): 'Y'})
        'Y'
    """

    # Initialisation du metteur à l'échelle par ligne
    def __init__(self, stage_scaler: StageScaler, layout: FrequencyLayout) -> None:
        self.stage_scaler = stage_scaler
        self.layout = layout

    # Méthode de liaison de fréquence d'une couche
    def layer_binding(self, layer: Layer) -> StageFrequency:
        """Shape the frequencies of a layer the way :class:`StageScaler` reads them.

        A per-entity mapping needs a grid carrying an entity level; a time
        series has none, and its single degenerate block is handed over as
        the scalar frequency instead.

        Args:
            layer: Mapping entity -> frequency, never empty.

        Returns:
            A copy of the mapping on a panel, the single frequency of the
            layer on a time series.

        Examples:
            >>> from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
            >>> layout = FrequencyLayout(
            ...     is_panel=True, detected_frequencies={}, undetected=(), misaligned=(),
            ...     target_frequency={('FR',): 'M'}, categories={}, progression=[],
            ... )
            >>> RowScaler(StageScaler(), layout).layer_binding({('FR',): 'Y'})
            {('FR',): 'Y'}
        """
        # Grille sans niveau d'entité : forme scalaire obligatoire
        if not self.layout.is_panel:
            return next(iter(layer.values()))
        return dict(layer)

    # Méthode des diviseurs de la cible
    def target_divisors(
        self,
        rows: TrainingRows,
        *,
        column: str,
        stage_freq: StageFrequency,
    ) -> pd.Series:
        """Divide the target by the period its own row spans.

        :meth:`StageScaler.target_divisor` with ``produced_freq``, called
        layer by layer on the real sub-grid of the layer.

        Args:
            rows: Decomposed rows of the training set.
            column: Column being imputed, for scale-mode resolution.
            stage_freq: Frequency of the stage.

        Returns:
            A ``Series`` of per-row divisors indexed like
            ``rows.row_frequency`` (all ``1.0`` when the target is exempt
            from scaling; NaN on a row of no layer).
        """
        # Fréquences de ligne, lues par position
        frequencies = rows.row_frequency.array

        # Un appel par couche, sur la grille réelle
        divisors = pd.Series(np.nan, index=rows.row_frequency.index, dtype=float)
        for layer, in_layer in rows.layers:
            positions = np.flatnonzero(in_layer)
            sub_real = rows.real_index[in_layer]
            produced = pd.Series(frequencies[in_layer], index=sub_real)
            sub_divisors = self.stage_scaler.target_divisor(
                column,
                source_freq=self.layer_binding(layer),
                pred_freq=stage_freq,
                index=sub_real,
                produced_freq=produced,
            )
            divisors.iloc[positions] = np.asarray(sub_divisors, dtype=float)
        return divisors

    # Méthode des diviseurs des covariables
    def feature_divisors(
        self,
        rows: TrainingRows,
        *,
        feature_cols: Sequence[str],
        freqs_by_column: ColumnFrequencies,
        ways: Mapping[str, MaterializationWay],
        stage_freq: StageFrequency,
    ) -> pd.DataFrame:
        """Divide the covariates by the period their own row spans.

        The covariates of a row produced at an earlier, finer stage are
        materialized at the row frequency (:class:`TrainingSetBuilder`), so
        their divisor is read there too — the very same rule the target
        divisor follows. :meth:`StageScaler.feature_divisors` receives the
        layer as ``grid_freq``, never the blocks.

        Args:
            rows: Decomposed rows of the training set.
            feature_cols: Covariate columns to scale.
            freqs_by_column: Detected frequencies, keyed by column.
            ways: Materialization way retained for each covariate.
            stage_freq: Frequency of the stage.

        Returns:
            A ``DataFrame`` of per-row divisors, indexed like
            ``rows.row_frequency`` and columned like ``feature_cols``: the
            per-row shape alone can carry two scales on one column.
        """
        # Diviseurs par ligne, NaN hors de toute couche
        divisors = pd.DataFrame(
            np.nan, index=rows.row_frequency.index, columns=list(feature_cols), dtype=float
        )
        for layer, in_layer in rows.layers:
            # Écriture positionnelle : sous une grille estampillée, deux lignes
            # partagent une date et l'alignement par étiquette les confondrait
            positions = np.flatnonzero(in_layer)
            sub_real = rows.real_index[in_layer]
            sub_divisors = self.stage_scaler.feature_divisors(
                columns=feature_cols,
                column_frequencies=freqs_by_column,
                ways=ways,
                grid_freq=self.layer_binding(layer),
                stage_freq=stage_freq,
                index=sub_real,
            )
            # Forme scalaire par colonne : diffusion sur les lignes concernées
            if isinstance(sub_divisors, pd.Series):
                for name in feature_cols:
                    divisors.iloc[
                        positions, divisors.columns.get_loc(name)
                    ] = float(sub_divisors[name])
            else:
                divisors.iloc[positions, :] = (
                    sub_divisors[list(feature_cols)].to_numpy(dtype=float)
                )
        return divisors
