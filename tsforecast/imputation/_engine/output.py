"""Output assembly of a ``HighFrequencyImputer`` pass, and the moves of its inversion.

Assembly: ``stage_output`` builds the frame and the provenance of one replayed stage (grid in input
order, imputations laid over, cells described as the frame carries them), ``assemble_output`` turns
the stage frames into the output of the pass — target level, multi-frequency stack
(``build_multifreq_output``) or prediction-pass rows — and ``split_target`` separates ``y``.
Inversion: the frequency level to keep and its removal, the mask of the ``ORIGINAL`` cells, the
inversion of the additive transformer and the restoration of the observed values.
"""
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Sklearn
from sklearn.base import TransformerMixin

# Utilitaires du package
from ...panel.utils import factorize_row_keys, get_entity_codes, normalize_entity_key
from ...utils._constants import FREQUENCY_LEVEL_NAME
from ...utils.frequency.utils import is_higher_frequency
# Composants de l'imputation
from .._components.covariate_materializer import CovariateMaterializer
from .._components.provenance import PROVENANCE_DTYPE, ProvenanceType, provenance_statistics
from .frequency_layout import (
    EntityKey,
    FrequencyLayout,
    PairKey,
    StageFrequency,
    stage_frequency_label,
    stage_frequency_of,
)
from .run_context import _RunContext

# Frames (ou provenances) d'étape, par label d'étape dans l'ordre de la progression
StageFrames = OrderedDict[str, pd.DataFrame]


# =============================================================================
# Frame et provenance d'une étape rejouée
# =============================================================================
# Fonction de la liaison de fréquence de la sortie d'une étape
def output_binding(
    binding: StageFrequency,
    carried: Mapping[EntityKey, str],
    index: pd.Index,
    layout: FrequencyLayout,
) -> StageFrequency:
    """Complete a stage binding with the entities the stage leaves out.

    A stage of a panel binds only the entities it concerns: the
    others have either already reached their target frequency or not yet
    entered the progression. The output, however, has to carry every
    entity, so each of them is bound to the frequency of its last stage —
    its target frequency when it has not travelled any stage yet, the
    frequency of the stage as a last resort: an entity without frequency
    would drop out of the output grid, hence out of the result.

    Args:
        binding: Execution binding of the stage, extended to the entities
            born at transform time.
        carried: Frequency of the last stage each entity took part in.
        index: Index of the working frame, source of the entity list.
        layout: Frequency layout of the fit, holder of the target frequency
            of each entity.

    Returns:
        The binding itself on a time series; a mapping naming every entity
        of the data otherwise.

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
        >>> layout = FrequencyLayout(
        ...     is_panel=True, detected_frequencies={}, undetected=(), misaligned=(),
        ...     target_frequency={('FR',): 'M', ('DE',): 'Q'}, categories={}, progression=[],
        ... )
        >>> index = pd.MultiIndex.from_product(
        ...     [['FR', 'DE'], pd.date_range('2021-01-31', periods=2, freq='ME')]
        ... )
        >>> output_binding({('FR',): 'M'}, {}, index, layout)
        {('FR',): 'M', ('DE',): 'Q'}
        >>> output_binding('M', {}, index, layout)
        'M'
    """
    # Série temporelle : la forme scalaire est la seule qui ait un sens
    if not isinstance(binding, dict):
        return binding

    # Parcours par entité
    completed = dict(binding)
    for entity in get_entity_codes(index)[1]:
        if entity in completed:
            continue
        # Report du dernier passage, à défaut la cible de l'entité, à
        # défaut encore la fréquence de l'étape : une entité sans
        # fréquence sortirait de la grille de sortie, donc du résultat
        frequency = (
            carried.get(entity)
            or layout.entity_target_frequency(entity)
            or next(iter(binding.values()), None)
        )
        if frequency is not None:
            completed[entity] = frequency
    return completed


# Fonction de mise dans l'ordre de l'entrée d'une grille d'étape
def in_input_order(grid: pd.Index, reference: pd.Index) -> pd.Index:
    """Order a stage grid like the input: entity blocks in input order, dates sorted.

    The window masks are concatenated entity by entity — and an entity
    the calculator omits is appended last —, so their index carries no
    guaranteed order. The output must keep the order of its input:
    an unsorted input is refused, and a component cannot produce what it
    refuses.

    Args:
        grid: Grid of a stage frame.
        reference: Index of the working frame, whose entity order wins.

    Returns:
        The grid, reordered.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.to_datetime(['2021-02-28', '2021-01-31'])
        >>> grid = pd.MultiIndex.from_arrays([['FR', 'FR'], dates])
        >>> reference = pd.MultiIndex.from_arrays([['DE', 'FR'], dates])
        >>> [str(date.date()) for date in in_input_order(grid, reference).get_level_values(-1)]
        ['2021-01-31', '2021-02-28']
        >>> in_input_order(pd.DatetimeIndex(dates), pd.DatetimeIndex(dates))[0].day
        31
    """
    # Série temporelle : l'ordre des dates suffit
    if not isinstance(grid, pd.MultiIndex):
        return grid.sort_values()

    # Panel : rang de chaque entité dans l'entrée, puis date
    rank = {
        entity: position
        for position, entity in enumerate(get_entity_codes(reference)[1])
    }
    levels = [grid.get_level_values(level) for level in range(grid.nlevels - 1)]
    entity_rank = np.asarray([
        rank.get(normalize_entity_key(tuple(values)), len(rank))
        for values in zip(*levels)
    ])
    order = np.lexsort((grid.get_level_values(-1).to_numpy(), entity_rank))
    return grid[order]


# Fonction de la frame et de la provenance d'une étape rejouée
def stage_output(
    run: _RunContext,
    layout: FrequencyLayout,
    X_stage: pd.DataFrame,
    grid: pd.Index,
    complete_binding: StageFrequency,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the output frame of one replayed stage, and its provenance.

    The stage frame is rebuilt after the writes of the stage on its grid —
    the index of the window mask, never its true rows alone —, then the
    imputations of the mirror are laid over it (:func:`overlay_imputations`):
    ``stage_frame`` reads the mirror only under
    ``covariate_strategy='model'``, and the output must carry it under every
    strategy. The provenance is a snapshot of the tracker taken at the same
    instant, reindexed on the grid and made to describe the cells of the
    frame (:func:`stage_provenance`): the matrix thus shares the index of the
    output, which the inversion reads mask against mask.

    Args:
        run: Execution context of the replay: materializer and registers,
            provenance tracker, frequencies of the pairs the fit never
            observed.
        layout: Frequency layout of the fit.
        X_stage: Working frame after the additive transformer.
        grid: Grid of the stage, in input order.
        complete_binding: Binding naming every entity of the data
            (:func:`output_binding`).

    Returns:
        Tuple ``(frame, provenance)`` on ``grid``, under the index names of
        ``X_stage`` when the grid has its depth.
    """
    # Frame d'étape reconstruite après les écritures
    frame = run.materializer.stage_frame(
        grid_index=grid,
        stage_freq=complete_binding,
        detected_frequencies=layout.frequencies_by_column(run.transform_frequencies),
        source_data=X_stage,
    )
    # Noms de l'index d'entrée : la grille vient d'un masque converti,
    # qui ne les porte pas, et la sortie doit rester lisible comme
    # l'entrée — empilée ou non
    if grid.nlevels == X_stage.index.nlevels:
        frame = frame.rename_axis(X_stage.index.names)
    # Recouvrement par les imputations, sous toutes les stratégies
    overlaid = overlay_imputations(run.materializer, frame, complete_binding)

    # Instantané de provenance, pris au même instant et réindexé sur la grille
    snapshot = run.provenance_tracker.get_provenance_matrix()
    provenance = stage_provenance(
        layout,
        run.transform_frequencies,
        snapshot.reindex(index=grid, columns=frame.columns),
        overlaid,
        complete_binding,
    ).rename_axis(frame.index.names)
    return overlaid, provenance


# Fonction de la provenance d'une frame d'étape
def stage_provenance(
    layout: FrequencyLayout,
    transform_frequencies: Mapping[PairKey, str],
    provenance: pd.DataFrame,
    frame: pd.DataFrame,
    complete_binding: StageFrequency,
) -> pd.DataFrame:
    """Make the tracker's marks describe the cells of one stage frame.

    The tracker marks the cells of the input grid, and a stage frame does
    not always carry the input value of a cell: a column observed more
    finely than the row is the exact **aggregation** of its period
    there, marked ``AGGREGATED`` — ``ORIGINAL`` would let the inversion
    hand a quarterly sum back as the monthly observation of the same
    date. A cell with a value no mark covers — an aggregation on a date
    absent from the input — is ``AGGREGATED`` too when the column is
    finer, ``ORIGINAL`` otherwise. And a cell left empty is never marked,
    whatever the tracker says of the input cell of the same date.

    Args:
        layout: Frequency layout of the fit.
        transform_frequencies: Frequencies the replay detected for the
            pairs the fit never observed.
        provenance: Snapshot of the tracker, reindexed on the frame.
        frame: Stage frame, imputations laid over.
        complete_binding: Complete frequency binding of the frame's grid.

    Returns:
        The provenance of the frame, cell for cell.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
        >>> layout = FrequencyLayout(
        ...     is_panel=False, detected_frequencies={'q1': 'Q'}, undetected=(),
        ...     misaligned=(), target_frequency='Y', categories={}, progression=[],
        ... )
        >>> index = pd.to_datetime(['2021-12-31', '2022-12-31'])
        >>> frame = pd.DataFrame({'q1': [10.0, np.nan]}, index=index)
        >>> marks = pd.DataFrame({'q1': [None, 'original']}, index=index)
        >>> stage_provenance(layout, {}, marks, frame, 'Y')['q1'].tolist()
        ['aggregated', nan]
    """
    # Extraction des valeurs renseignées
    has_value = frame.notna()
    # Colonnes que le suivi ne porte pas : ajoutées par la réindexation en
    # float, ramenées au dtype catégoriel avant tout marquage
    provenance = provenance.astype(PROVENANCE_DTYPE)

    # Cellules sans marque mais renseignées : valeurs reprises telles quelles
    provenance = provenance.mask(provenance.isna() & has_value, ProvenanceType.ORIGINAL)
    # Agrégations exactes d'une colonne plus fine que la ligne
    aggregated = (
        aggregated_cells(layout, transform_frequencies, frame, complete_binding)
        & has_value
    )
    provenance = provenance.mask(aggregated, ProvenanceType.AGGREGATED)
    # Cellule vide : jamais marquée, au rejeu comme au fit
    return provenance.where(has_value)


# Fonction des cellules agrégées d'une frame d'étape
def aggregated_cells(
    layout: FrequencyLayout,
    transform_frequencies: Mapping[PairKey, str],
    frame: pd.DataFrame,
    complete_binding: StageFrequency,
) -> pd.DataFrame:
    """Locate the cells of a stage frame whose column is finer than the row.

    Such a column is never imputed at that stage: the stage frame
    carries its exact additive aggregation over the period of the row.

    Args:
        layout: Frequency layout of the fit.
        transform_frequencies: Frequencies the replay detected for the
            pairs the fit never observed.
        frame: Stage frame.
        complete_binding: Complete frequency binding of the frame's grid.

    Returns:
        Boolean frame shaped like ``frame``.

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
        >>> layout = FrequencyLayout(
        ...     is_panel=False, detected_frequencies={'m1': 'M', 'a1': 'Y'}, undetected=(),
        ...     misaligned=(), target_frequency='Q', categories={}, progression=[],
        ... )
        >>> frame = pd.DataFrame({'m1': [1.0], 'a1': [2.0]}, index=pd.to_datetime(['2021-03-31']))
        >>> aggregated_cells(layout, {}, frame, 'Q').iloc[0].tolist()
        [True, False]
    """
    # Fréquence d'étape et entité de chaque ligne, codées en couples
    # distincts sans construire de tuple par ligne
    row_frequency = stage_frequency_of(complete_binding, frame.index)
    entity_codes, entities = get_entity_codes(frame.index)
    codes, first_rows = factorize_row_keys(entity_codes, row_frequency)
    pairs = [
        (entities[entity_codes[row]], row_frequency.iat[row]) for row in first_rows
    ]

    # Fréquence de chaque colonne, par entité sur un panel
    column_frequencies = layout.frequencies_by_column(transform_frequencies)
    result = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    # Détection des colonnes à plus haute fréquence que celle de l'étape
    for column in frame.columns:
        frequencies = column_frequencies.get(column)
        if frequencies is None:
            continue
        finer = []
        for entity, stage in pairs:
            own = (
                frequencies.get(entity) if isinstance(frequencies, dict)
                else frequencies
            )
            finer.append(own is not None and is_higher_frequency(own, stage))
        result[column] = np.asarray(finer, dtype=bool)[codes]
    return result


# Fonction de recouvrement d'une frame par les imputations
def overlay_imputations(
    materializer: CovariateMaterializer,
    frame: pd.DataFrame,
    complete_binding: StageFrequency,
) -> pd.DataFrame:
    """Lay the imputations of the mirror over a stage frame.

    ``stage_frame`` fabricates nothing: it carries the input values and
    the exact aggregations, and reads the mirror only under
    ``covariate_strategy='model'``. The output must show the imputations
    under every strategy, and they must win over the raw anchor of a
    lower-frequency column — an anchor row re-expressed no longer carries
    the total of its period.

    Only the cells produced at the frequency of the row are laid over: a
    value produced at an earlier, coarser stage would otherwise land on a
    finer grid at the magnitude of another period.

    Args:
        materializer: Materializer of the replay, holder of the mirror.
        frame: Stage frame, as built by the materializer.
        complete_binding: Complete frequency binding of the frame's grid.

    Returns:
        The frame, imputations included.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tsforecast.imputation._components.covariate_materializer import (
        ...     CovariateMaterializer,
        ... )
        >>> index = pd.date_range('2021-01-31', periods=3, freq='ME')
        >>> materializer = CovariateMaterializer()
        >>> materializer.imputed_store['q1'] = pd.Series([1.0, 2.0, 3.0], index=index)
        >>> materializer.imputed_freq_store['q1'] = pd.Series(['M', 'Q', 'M'], index=index)
        >>> frame = pd.DataFrame({'q1': [np.nan, np.nan, 9.0]}, index=index)
        >>> overlay_imputations(materializer, frame, 'M')['q1'].tolist()
        [1.0, nan, 3.0]
    """
    # Aucune production : la frame est déjà la sortie de l'étape
    if not materializer.imputed_store:
        return frame

    # Fréquence d'étape de chaque ligne de la grille
    row_frequency = stage_frequency_of(complete_binding, frame.index)

    # Initialisation du jeu de données résultat
    result = frame.copy()
    # Parcours des colonnes
    for column in result.columns:
        # Cellules imputées
        produced = materializer.imputed_store.get(column)
        if produced is None:
            continue
        values = produced.reindex(frame.index)
        # Fréquence de production, cellule par cellule
        produced_frequency = materializer.imputed_freq_store[column].reindex(
            frame.index
        )
        # Cellules de l'étape courante, les seules à la bonne échelle
        kept = values.where(
            values.notna() & produced_frequency.eq(row_frequency)
        )
        result[column] = kept.combine_first(result[column])
    return result


# =============================================================================
# Sortie d'une passe
# =============================================================================
# Fonction d'empilage des niveaux de fréquence
def build_multifreq_output(stage_frames: StageFrames, *, is_panel: bool) -> pd.DataFrame:
    """Stack the stage frames into one multi-frequency frame.

    Backs ``keep_lower_frequencies=True``, a **pure display parameter**: it
    governs how the levels of the output are stacked, never the logic. The
    index adds the frequency level that sits on the
    entity side and every entity level of a panel is preserved with its
    own name.

    Under ``impute_intermediate_frequencies=False`` the progression holds
    the target stage alone: the output then carries that single level,
    there being no intermediate one to stack.

    Args:
        stage_frames: Stage frames keyed by frequency label, in progression
            order. Each must already carry the imputations of its stage —
            :func:`stage_output` builds them after the writes.
        is_panel: Whether the data are a panel.

    Returns:
        Frame with a MultiIndex ``(frequency, date)`` on a time series and
        ``(entity..., 'frequency', 'date')`` on a panel.

    Examples:
        >>> from collections import OrderedDict
        >>> import pandas as pd
        >>> quarterly = pd.DataFrame({'v': [1.0]}, index=pd.to_datetime(['2021-03-31']))
        >>> monthly = pd.DataFrame({'v': [2.0, 3.0]}, index=pd.to_datetime(['2021-01-31', '2021-02-28']))
        >>> stacked = build_multifreq_output(OrderedDict(Q=quarterly, M=monthly), is_panel=False)
        >>> stacked.index.names, stacked.index.get_level_values(0).tolist()
        (FrozenList(['frequency', 'date']), ['Q', 'M', 'M'])
    """
    # Empilage dans l'ordre d'insertion, l'étiquette de chaque bloc étant
    # construite à part pour ne jamais entrer en collision avec une colonne
    all_frames = list(stage_frames.values())
    combined = pd.concat(all_frames, ignore_index=False)
    # Niveau de fréquence en dtype chaîne, quel que soit le nombre d'étapes :
    # l'étiquette de chaque étape répétée sur ses lignes, par position
    labels = pd.array(list(stage_frames), dtype='str')
    freq_values = pd.Index(
        labels.take(np.repeat(np.arange(len(labels)), [len(frame) for frame in all_frames]))
    )

    # Construction du MultiIndex de sortie
    if is_panel and isinstance(combined.index, pd.MultiIndex):
        # Conservation de tous les niveaux d'entité, noms compris
        n_entity = combined.index.nlevels - 1
        entity_arrays = [
            combined.index.get_level_values(level) for level in range(n_entity)
        ]
        entity_names = [
            combined.index.names[level]
            if combined.index.names[level] is not None
            else ('entity' if n_entity == 1 else f'entity_{level}')
            for level in range(n_entity)
        ]
        new_index = pd.MultiIndex.from_arrays(
            [*entity_arrays, freq_values, combined.index.get_level_values(-1)],
            names=[
                *entity_names,
                FREQUENCY_LEVEL_NAME,
                combined.index.names[-1] or 'date',
            ],
        )
    else:
        new_index = pd.MultiIndex.from_arrays(
            [freq_values, combined.index],
            names=[FREQUENCY_LEVEL_NAME, combined.index.name or 'date'],
        )

    return combined.set_axis(new_index)


# Sortie d'une passe de transform
@dataclass(frozen=True)
class PassOutput:
    """Output of one transform pass, before the split of ``X`` and ``y``.

    Attributes:
        data: Transformed frame: the target level, the multi-frequency stack,
            or the input rows of the prediction pass.
        provenance: Provenance matrix of ``data``, cell for cell.
        statistics: Provenance statistics, counted on the target level.
        stacked_nlevels: Depth of the stacked index, None when the output is
            not stacked; it identifies a multi-frequency frame at inversion.
    """

    data: pd.DataFrame
    provenance: pd.DataFrame
    statistics: Dict[str, Dict[str, Union[int, float]]]
    stacked_nlevels: Optional[int] = None


# Fonction de la sortie d'une passe, à partir des frames d'étape
def assemble_output(
    frames: StageFrames,
    provenances: StageFrames,
    *,
    prediction: bool,
    keep_lower_frequencies: bool,
    is_panel: bool,
    input_index: pd.Index,
    input_columns: pd.Index,
) -> PassOutput:
    """Turn the stage frames of a replay into the output of the pass.

    The progression always ends on the target stage: the last frame is the
    target level, whatever the shape of the output.

    - prediction pass: the target level reindexed on the input rows and
      the columns of ``X`` — no row added, no frequency level;
    - ``keep_lower_frequencies=True``: every stage stacked on a frequency
      level, the provenance alike;
    - otherwise: the target level alone.

    The statistics always describe the target level: ``keep_lower_frequencies``
    is a display parameter only.

    Args:
        frames: Stage frames, keyed by label in progression order.
        provenances: Their provenance matrices, same keys.
        prediction: Whether the pass is the prediction pass.
        keep_lower_frequencies: The display parameter of the imputer.
        is_panel: Whether the data are a panel.
        input_index: Index of the input ``X``, the rows of the prediction pass.
        input_columns: Columns of the input ``X``, those of the prediction pass.

    Returns:
        The :class:`PassOutput` of the pass.

    Examples:
        >>> from collections import OrderedDict
        >>> import pandas as pd
        >>> index = pd.to_datetime(['2021-01-31', '2021-02-28'])
        >>> frame = pd.DataFrame({'v': [1.0, 2.0]}, index=index)
        >>> marks = pd.DataFrame({'v': ['original', 'model_on_true']}, index=index)
        >>> output = assemble_output(
        ...     OrderedDict(M=frame), OrderedDict(M=marks), prediction=False,
        ...     keep_lower_frequencies=False, is_panel=False,
        ...     input_index=index, input_columns=frame.columns,
        ... )
        >>> output.data is frame, output.stacked_nlevels, output.statistics['v']['original']
        (True, None, 1)
    """
    # Niveau cible : la dernière frame de la progression
    final_label = list(frames)[-1]
    # Profondeur de l'index empilé, notée au moment de l'empilement : c'est
    # elle, et non le nom d'un niveau, qui identifie à l'inversion une
    # sortie multi-fréquences
    stacked_nlevels: Optional[int] = None
    if prediction:
        # Passe de prédiction : niveau cible, sur les lignes d'entrée et les
        # colonnes de X — aucune ligne ajoutée, ni niveau de fréquence
        data = frames[final_label].reindex(index=input_index, columns=input_columns)
        # Colonne absente de l'étape : réindexée en float, ramenée au dtype catégoriel
        provenance = provenances[final_label].reindex(
            index=input_index, columns=input_columns
        ).astype(PROVENANCE_DTYPE)
    elif keep_lower_frequencies:
        data = build_multifreq_output(frames, is_panel=is_panel)
        provenance = build_multifreq_output(provenances, is_panel=is_panel)
        stacked_nlevels = data.index.nlevels
    else:
        data = frames[final_label]
        provenance = provenances[final_label]

    # Statistiques du niveau cible
    statistics = provenance_statistics(
        provenance if prediction else provenances[final_label]
    )
    return PassOutput(
        data=data,
        provenance=provenance,
        statistics=statistics,
        stacked_nlevels=stacked_nlevels,
    )


# Fonction de concaténation de X et de la cible
def merge_target(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    target_column: Optional[str],
) -> pd.DataFrame:
    """Merge the target into the features under its working-frame name.

    Args:
        X: Features.
        y: Target, optional.
        target_column: Name the target takes in the frame.

    Returns:
        A new frame: ``X`` and the target column, or a copy of ``X`` alone.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1.0]})
        >>> list(merge_target(X, pd.Series([2.0], name='y'), 'y').columns)
        ['a', 'y']
    """
    # Sans cible : copie, jamais l'objet reçu
    if y is None:
        return X.copy()
    return pd.concat([X, y.to_frame(name=target_column)], axis=1)


# Fonction de scission X / y d'une sortie
def split_target(
    data: pd.DataFrame,
    y: Optional[pd.Series],
    target_column: Optional[str],
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Split the target off an output, under the name it was given.

    Args:
        data: Output frame, the target column included when ``y`` was merged.
        y: Target as given, read for its name; None without one.
        target_column: Name of the target in ``data``.

    Returns:
        The pair ``(X, y)`` when ``y`` is given and ``data`` carries its
        column, ``data`` itself otherwise.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'a': [1.0], '__target__': [2.0]})
        >>> X, y = split_target(data, pd.Series([0.0]), '__target__')
        >>> list(X.columns), y.name
        (['a'], None)
    """
    # La cible garde le nom qu'elle portait en entrée
    if y is not None and target_column in data.columns:
        return (
            data.drop(columns=[target_column]),
            data[target_column].rename(y.name),
        )
    return data


# =============================================================================
# Inversion
# =============================================================================
# Fonction de présence du niveau de fréquence
def has_frequency_level(frame: pd.DataFrame, stacked_nlevels: Optional[int]) -> bool:
    """Tell whether a frame is a stacked multi-frequency frame.

    Read on the STRUCTURE, never on the level names: a stacked frame is
    a MultiIndex as deep as the output the last transform stacked, its
    frequency level sitting just before the date. Renaming the levels of
    the output, or an index level named like the frequency level, cannot
    fool the test.

    Args:
        frame: Frame to inspect.
        stacked_nlevels: Depth of the output the last transform stacked,
            None when it did not stack it.

    Returns:
        True if the last transform stacked its output and ``frame`` has
        that output's depth.

    Examples:
        >>> import pandas as pd
        >>> stacked = pd.DataFrame(
        ...     {'v': [1.0]},
        ...     index=pd.MultiIndex.from_tuples([('M', pd.Timestamp('2021-01-31'))]),
        ... )
        >>> has_frequency_level(stacked, 2), has_frequency_level(stacked, None)
        (True, False)
    """
    return (
        stacked_nlevels is not None
        and isinstance(frame.index, pd.MultiIndex)
        and frame.index.nlevels == stacked_nlevels
    )


# Fonction de sélection du niveau de fréquence à inverser
def select_inverse_frequency_level(
    data: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    stacked_nlevels: Optional[int],
    source_label: Optional[str],
    target_frequency: StageFrequency,
) -> Optional[str]:
    """Pick the frequency level to keep when inverting a stacked output.

    The level of the source index is the one to keep: it is where the
    values sit at the granularity of the data given to ``fit``. It may be
    missing — an undetectable index frequency, or a frequency that never
    was a stage — and the target level is then the best proxy, being the
    only one always produced.

    The levels are read on the data to invert whenever they are stacked:
    a level only the provenance carries would select nothing in them. The
    provenance is read only for flat data.

    Args:
        data: Frame to invert, stacked or not.
        provenance: Provenance matrix of the last transform, stacked or not
            — both follow ``keep_lower_frequencies``.
        stacked_nlevels: Depth of the output the last transform stacked.
        source_label: Frequency label of the input index of the fit, None
            when it could not be detected.
        target_frequency: Effective target frequency of the fit.

    Returns:
        The frequency label to keep, or None when neither frame carries a
        frequency level.

    Examples:
        >>> import pandas as pd
        >>> stacked = pd.DataFrame({'v': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples(
        ...     [('Q', pd.Timestamp('2021-03-31')), ('M', pd.Timestamp('2021-03-31'))]
        ... ))
        >>> select_inverse_frequency_level(
        ...     stacked, stacked, stacked_nlevels=2, source_label=None, target_frequency='M'
        ... )
        'M'
    """
    # Labels disponibles : ceux des données, à défaut ceux de la provenance
    frame = data if has_frequency_level(data, stacked_nlevels) else provenance
    if not has_frequency_level(frame, stacked_nlevels):
        return None
    # Niveau de fréquence lu par position : l'avant-dernier
    available = frame.index.get_level_values(-2).unique().tolist()

    # Priorité au niveau de l'index source
    if source_label is not None and source_label in available:
        return source_label

    # Repli sur le niveau cible, toujours produit
    target_label = stage_frequency_label(target_frequency)
    if target_label in available:
        return target_label

    # Dernier repli : le dernier niveau empilé, avec avertissement
    warnings.warn(
        f"Neither the source frequency level ({source_label}) nor the "
        f"target one ({target_label}) is present in the data to invert. "
        f"Falling back on the last stacked level {available[-1]!r}.",
        UserWarning,
    )
    return available[-1]


# Fonction de suppression du niveau de fréquence
def drop_frequency_level(
    frame: pd.DataFrame,
    label: Optional[str],
    *,
    stacked_nlevels: Optional[int],
    index_names: Sequence[Any],
) -> pd.DataFrame:
    """Reduce a stacked frame to one frequency level and restore its index.

    Panel and time series are handled alike: the level is addressed by
    position — the one before the date —, never by name, and the names of
    the remaining levels are restored from the last transform input.

    Args:
        frame: Frame to reduce, stacked or not.
        label: Frequency label to keep. None leaves the frame untouched.
        stacked_nlevels: Depth of the output the last transform stacked.
        index_names: Index names of the input of the last pass, which has
            by construction the depth of the reduced frame.

    Returns:
        The frame restricted to ``label``, without the frequency level.

    Raises:
        ValueError: If ``frame`` is stacked but carries no ``label`` level
            — the provenance of the last transform lacking a level the
            data to invert hold: those data are not an output of that
            transform.

    Examples:
        >>> import pandas as pd
        >>> stacked = pd.DataFrame({'v': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples(
        ...     [('Q', pd.Timestamp('2021-03-31')), ('M', pd.Timestamp('2021-03-31'))]
        ... ))
        >>> reduced = drop_frequency_level(stacked, 'M', stacked_nlevels=2, index_names=['date'])
        >>> reduced.index.name, reduced['v'].tolist()
        ('date', [2.0])
    """
    # Frame déjà à un seul niveau de fréquence
    if label is None or not has_frequency_level(frame, stacked_nlevels):
        return frame

    # Niveau de fréquence : l'avant-dernier, par position. Un niveau
    # absent ne laisse rien à inverser : une frame empilée et une frame
    # réduite ne s'alignent sur aucune cellule
    level = frame.index.nlevels - 2
    if label not in frame.index.get_level_values(level):
        raise ValueError(
            f"The frequency level {label!r} of the data to invert is not "
            f"one the last transform produced: inverse_transform undoes "
            f"the last pass, so it inverts an output of that pass only."
        )
    reduced = frame.xs(label, level=level)

    # Noms de niveaux de l'index source
    return reduced.rename_axis(index_names)


# Fonction de remise à NaN des cellules non originales
def keep_original_cells(data: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    """Set back to NaN every cell whose provenance is not ``ORIGINAL``.

    A cell the provenance does not cover is not original; a column outside
    the provenance — a column the imputer never tracked — is never masked.

    Args:
        data: Frame to invert, at the index of the provenance.
        provenance: Provenance matrix of the last pass, reduced to one level.

    Returns:
        A new frame, non-original cells set to NaN.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'v': [1.0, 2.0], 'extra': [3.0, 4.0]})
        >>> marks = pd.DataFrame({'v': ['original', 'model_on_true']})
        >>> keep_original_cells(data, marks)['v'].tolist()
        [1.0, nan]
    """
    # Masque des cellules originales, aligné sur les données
    original_mask = (provenance == ProvenanceType.ORIGINAL).reindex(
        index=data.index, columns=data.columns, fill_value=False
    )
    # Colonnes hors périmètre de la provenance : jamais masquées
    untracked = [c for c in data.columns if c not in provenance.columns]
    if untracked:
        original_mask[untracked] = True
    return data.where(original_mask)


# Fonction d'inversion du transformateur additif
def invert_additive_transformer(
    transformer: Optional[TransformerMixin],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Undo the additive transformer, last move of the inversion.

    Mirror of the transform, which applies it first. A transformer without
    ``inverse_transform`` leaves the data as they are; an inversion that
    fails warns and leaves them too.

    Args:
        transformer: Fitted additive transformer, None without one.
        data: Frame to invert, back on the input index.

    Returns:
        The inverted frame — the first element when the transformer returns
        an ``(X, y)`` pair — or ``data`` itself.

    Examples:
        >>> import pandas as pd
        >>> invert_additive_transformer(None, pd.DataFrame({'v': [1.0]}))['v'].tolist()
        [1.0]
    """
    # Sans transformateur inversible : rien à défaire
    if transformer is None or not hasattr(transformer, 'inverse_transform'):
        return data
    try:
        inverted = transformer.inverse_transform(data)
        return inverted[0] if isinstance(inverted, tuple) else inverted
    except Exception as error:
        warnings.warn(
            f"Failed to inverse transform with the additive "
            f"transformer: {error}",
            UserWarning,
        )
        return data


# Fonction de restauration des valeurs d'origine exactes
def restore_original_values(
    data: pd.DataFrame,
    snapshot_X: pd.DataFrame,
    snapshot_y: Optional[pd.Series],
    target_column: Optional[str],
) -> pd.DataFrame:
    """Refill the cells observed in the input of the last pass.

    The last pass is the last ``transform``, or the ``fit`` when no
    ``transform`` has followed it: both snapshot their input.

    Backs ``restore_original_values=True``: the ``ORIGINAL`` mask alone
    drops the anchor dates of a lower-frequency variable, which the target
    level holds as an imputation even though the input carried a true
    observation there.

    Args:
        data: Frame restored from the provenance mask, at the source index.
        snapshot_X: Snapshot of the features of the last pass.
        snapshot_y: Snapshot of its target when it was merged, else None.
        target_column: Column name given to ``y`` in the working frame.

    Returns:
        The frame with every cell observed in the snapshot set back to its
        original value.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({'v': [np.nan, 2.0]})
        >>> snapshot = pd.DataFrame({'v': [1.0, np.nan]})
        >>> restore_original_values(data, snapshot, None, None)['v'].tolist()
        [1.0, 2.0]
    """
    # Reconstruction de l'instantané de l'entrée de la dernière passe
    snapshot = snapshot_X
    if snapshot_y is not None and target_column is not None:
        snapshot = pd.concat(
            [snapshot, snapshot_y.to_frame(name=target_column)], axis=1
        )

    # Restriction aux colonnes et à l'index communs
    common_cols = [c for c in data.columns if c in snapshot.columns]
    if not common_cols:
        return data
    aligned = snapshot[common_cols].reindex(index=data.index)

    # Les valeurs observées priment sur celles restaurées par la provenance
    data = data.copy()
    data[common_cols] = aligned.combine_first(data[common_cols])
    return data
