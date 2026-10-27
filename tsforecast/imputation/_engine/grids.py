"""Grids and window masks of one ``HighFrequencyImputer`` stage.

Pure functions reading the window calculator of a pass on the index of the working frame:
``restrict_to_entities`` keeps the rows of a group of entities, ``stage_mask`` reads one window
mask at the frequency of a stage, ``prediction_grid`` and ``unrestricted_grid`` build the grid a
step writes on, and ``training_mask_at`` reads the training window injected into the training-set
builder.
"""
# Importation des modules
# Modules de base
from typing import Dict, Literal, Optional, Sequence

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Utilitaires du package
from ...panel.utils import get_entity_codes, normalize_entity_key
from .._components.imputation_window import ImputationWindowCalculator
from .frequency_layout import EntityKey, LogCallback, StageFrequency


# Fonction de restriction d'un index aux entités d'un groupe
def restrict_to_entities(
    index: pd.Index,
    entities: Optional[Sequence[EntityKey]],
) -> pd.Index:
    """Restrict an index to the rows of a group of entities.

    The membership is decided once per distinct entity, then broadcast to the
    rows through their integer entity codes: no key is built per row.

    Args:
        index: Index to restrict.
        entities: Entity keys kept, None keeping everything (time series).

    Returns:
        The restricted index, in its original order.

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=2, freq='ME')
        >>> index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        >>> restrict_to_entities(index, [('DE',)]).get_level_values('country').tolist()
        ['DE', 'DE']
        >>> restrict_to_entities(dates, None).equals(dates)
        True
    """
    # Série temporelle, ou groupe couvrant tout le panel
    if entities is None or not isinstance(index, pd.MultiIndex):
        return index

    # Appartenance décidée par entité distincte, diffusée aux lignes par leurs codes
    kept = {normalize_entity_key(entity) for entity in entities}
    codes, index_entities = get_entity_codes(index)
    selected = np.array([entity in kept for entity in index_entities], dtype=bool)
    return index[selected[codes]]


# Fonction de lecture d'un masque de fenêtre à la fréquence d'une étape
def stage_mask(
    window_calculator: ImputationWindowCalculator,
    index: pd.Index,
    stage_freq: StageFrequency,
    kind: Literal['imputation', 'training'],
    *,
    pass_name: str,
    log: Optional[LogCallback] = None,
) -> pd.Series:
    """Read one window mask at the frequency of a stage.

    The ``kind`` is named by the caller, never defaulted. Only the two
    windows the fit actually restricts on are read here: the strict
    window is diagnostic (``strict_window_mask_``) and governs nothing
    in the stage execution.
    An entity the calculator omits — one without a valid fitted mask — is
    left unrestricted rather than silently losing every one of its rows.

    Args:
        window_calculator: Window calculator fitted on the data of the pass.
        index: Index of the working frame, the fallback grid.
        stage_freq: Frequency of the stage, scalar or per entity.
        kind: Window read: ``'imputation'`` or ``'training'``.
        pass_name: Name of the pass, prefixed to the log message.
        log: Logging callback, or None.

    Returns:
        Boolean Series at the stage frequency, on a ``DatetimeIndex`` for
        a time series and on a ``(entity..., date)`` ``MultiIndex``
        otherwise.

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(6.0)}, index=dates)
        >>> calculator = ImputationWindowCalculator().fit(data)
        >>> stage_mask(calculator, data.index, 'QE', 'imputation', pass_name='fit').tolist()
        [True, True]
        >>> stage_mask(ImputationWindowCalculator(), data.index, 'QE', 'imputation',
        ...            pass_name='fit', log=print).all()
        [fit] imputation mask unavailable (Calculator not fitted. Call fit() first.); every row kept
        np.True_
    """
    # Calculateur non ajusté ou fréquence inconvertible : aucune
    # restriction, la garde de la phase 1 ayant déjà averti
    try:
        mask = window_calculator.get_mask_at_frequency(stage_freq, kind=kind)
    except (ValueError, KeyError, TypeError) as error:
        if log is not None:
            log(f"[{pass_name}] {kind} mask unavailable ({error}); every row kept")
        return pd.Series(True, index=index)

    # Entités omises par le calculateur : rattachées sans restriction
    covered = set(get_entity_codes(mask.index)[1])
    missing = [entity for entity in get_entity_codes(index)[1] if entity not in covered]
    if missing:
        extra = restrict_to_entities(index, missing)
        mask = pd.concat([mask, pd.Series(True, index=extra)])
    return mask


# Fonction de la grille de prédiction d'une étape
def prediction_grid(
    window_calculator: ImputationWindowCalculator,
    index: pd.Index,
    stage_freq: StageFrequency,
    entities: Optional[Sequence[EntityKey]],
    *,
    pass_name: str,
    log: Optional[LogCallback] = None,
) -> pd.Index:
    """Build the prediction grid of a stage, restricted to one group.

    The grid spans the whole imputation window of the entities, anchor
    rows included: a period holding no observation is
    predicted like any other.

    Args:
        window_calculator: Window calculator fitted on the data of the pass.
        index: Index of the working frame.
        stage_freq: Frequency of the stage.
        entities: Entities of the group, None for a time series.
        pass_name: Name of the pass, for the log message.
        log: Logging callback, or None.

    Returns:
        Index of the rows the step writes on.

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        >>> data = pd.DataFrame({'m1': [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        >>> calculator = ImputationWindowCalculator().fit(data)
        >>> prediction_grid(calculator, data.index, 'QE', None, pass_name='fit').strftime('%Y-%m').tolist()
        ['2021-06']
    """
    # Masque d'imputation, "kind" nommé explicitement
    mask = stage_mask(
        window_calculator, index, stage_freq, kind='imputation', pass_name=pass_name, log=log
    )
    grid = mask.index[mask.to_numpy(dtype=bool)]
    return restrict_to_entities(grid, entities)


# Fonction de la grille non restreinte d'un groupe
def unrestricted_grid(
    window_calculator: ImputationWindowCalculator,
    index: pd.Index,
    stage_freq: StageFrequency,
    entities: Optional[Sequence[EntityKey]],
    *,
    pass_name: str,
    log: Optional[LogCallback] = None,
) -> pd.Index:
    """Build the stage grid of a group, window restriction lifted.

    Reserved to the unanchored steps, whose entities have
    no window to speak of: the prediction window of such an entity is
    empty, since the imputed column carries no anchor to bound it. Every
    row of the stage grid is kept; the covariates then decide, a row they
    cannot feed producing a NaN that is simply never written.

    Args:
        window_calculator: Window calculator fitted on the data of the pass.
        index: Index of the working frame.
        stage_freq: Frequency of the stage.
        entities: Entities of the group, None for a time series.
        pass_name: Name of the pass, for the log message.
        log: Logging callback, or None.

    Returns:
        Index of the rows of those entities at the stage frequency.

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        >>> data = pd.DataFrame({'m1': [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        >>> calculator = ImputationWindowCalculator().fit(data)
        >>> unrestricted_grid(calculator, data.index, 'QE', None, pass_name='fit').strftime('%Y-%m').tolist()
        ['2021-03', '2021-06']
    """
    # Index du masque, pris sans sa valeur booléenne : la grille de l'étape
    mask = stage_mask(
        window_calculator, index, stage_freq, kind='imputation', pass_name=pass_name, log=log
    )
    return restrict_to_entities(mask.index, entities)


# Fonction du masque d'entraînement lu à la fréquence des blocs
def training_mask_at(
    window_calculator: ImputationWindowCalculator,
    frequencies: Dict[EntityKey, str],
    *,
    is_panel: bool,
    log: Optional[LogCallback] = None,
) -> Optional[pd.Series]:
    """Read the ``'training'`` window mask at one frequency per block.

    Injected into :class:`TrainingSetBuilder`, bound to the calculator of
    the fit: the builder therefore receives a callable and never the
    calculator, so it cannot read a window other than the training one,
    whose ``kind`` is named here once and for all.

    Args:
        window_calculator: Window calculator of the fit.
        frequencies: Mapping entity -> block frequency ``f_block(e)``.
        is_panel: Whether the data is a panel.
        log: Logging callback, or None.

    Returns:
        Boolean mask at those frequencies, on a ``DatetimeIndex`` for a
        time series and on a ``(entity..., date)`` MultiIndex otherwise;
        None when no window is computable — an unfitted calculator, or a
        block frequency it cannot convert to — which the builder reads as
        "no restriction at all".

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(6.0)}, index=dates)
        >>> calculator = ImputationWindowCalculator().fit(data)
        >>> training_mask_at(calculator, {(): 'QE'}, is_panel=False).tolist()
        [True, True]
        >>> training_mask_at(ImputationWindowCalculator(), {(): 'QE'}, is_panel=False) is None
        True
    """
    # Série temporelle : le calculateur attend la fréquence scalaire du
    # bloc unique, une liaison par entité n'ayant pas de sens sans entité
    binding: StageFrequency = frequencies
    if not is_panel and isinstance(frequencies, dict):
        binding = next(iter(frequencies.values()))
    try:
        return window_calculator.get_mask_at_frequency(binding, kind='training')
    except (ValueError, KeyError, TypeError) as error:
        # Logging
        if log is not None:
            log(f"[fit] training mask unavailable ({error}); every row kept")
        return None
