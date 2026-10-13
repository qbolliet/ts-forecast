"""Shared builders of the ``HighFrequencyImputer`` tests (not a test module).

Entity keys of ``PANEL-F``, a factory of valid imputers, quiet ``fit`` /
``transform`` wrappers, the spy-estimator fit, index helpers grouping dates by
entity, the six significant combinations of the two axes, and the PUBLIC
readers the suite uses instead of the imputer's private members: the fitting
context of each (stage, variable) caught on ``TrainingSetBuilder.build`` and
the plan models, the imputable groups read on the plan, and the imputed values
read on the ``transform`` output — this package is the safety net of the
refactoring, so it must not hold on to what the refactoring will move.
"""
# Modules de base
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple, Union
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LinearRegression

# Objet testé et collaborateurs publics
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.training_set_builder import TrainingSet, TrainingSetBuilder
from tsforecast.utils.frequency import normalize_frequency

# Estimateurs factices partagés
from tests.support.estimators import SpyEstimator

# Clés d'entité du jeu PANEL-F, sous forme de tuples (§2.5)
FR, DE, IT = ('FR',), ('DE',), ('IT',)

# Les six combinaisons significatives des deux axes : 'tolerate_nan' est
# exclue, son prérequis dur (un estimateur tolérant les NaN) relevant d'un
# autre test
AXIS_COMBINATIONS: List[Tuple[str, object]] = [
    ('interpolate', False),
    ('interpolate', 'covariates_only'),
    ('interpolate', True),
    ('model', False),
    ('model', 'covariates_only'),
    ('model', True),
]


# Fabrique d'un imputeur valide, surchargeable paramètre par paramètre
def make_imputer(**overrides) -> HighFrequencyImputer:
    """Build a valid imputer, overriding the given parameters.

    Args:
        **overrides: Constructor parameters replacing the defaults
            ``target_frequency='M'`` and ``estimator=LinearRegression()``.

    Returns:
        An unfitted imputer.

    Examples:
        >>> make_imputer(covariate_strategy='model').covariate_strategy
        'model'
    """
    params = dict(target_frequency='M', estimator=LinearRegression())
    params.update(overrides)
    return HighFrequencyImputer(**params)


# Ajustement silencieux
def fit_quietly(imputer: HighFrequencyImputer, data: pd.DataFrame) -> HighFrequencyImputer:
    """Fit the imputer, swallowing the warnings the fit legitimately emits.

    Args:
        imputer: Imputer to fit.
        data: Training data.

    Returns:
        The fitted imputer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(data)


# Rejeu silencieux
def transform_quietly(imputer: HighFrequencyImputer, data: pd.DataFrame) -> pd.DataFrame:
    """Replay the plan on data, swallowing the legitimate warnings.

    Args:
        imputer: Fitted imputer.
        data: Data to transform.

    Returns:
        The transformed frame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.transform(data)


# Ajustement puis rejeu silencieux
def fit_transform_quietly(imputer: HighFrequencyImputer, data: pd.DataFrame) -> pd.DataFrame:
    """Fit then replay, swallowing the legitimate warnings.

    Args:
        imputer: Imputer to fit.
        data: Training data, transformed too.

    Returns:
        The transformed frame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit_transform(data)


# Ajustement silencieux avec l'estimateur espion
def fit_with_spy(data: pd.DataFrame, **overrides) -> HighFrequencyImputer:
    """Fit an imputer carrying the spy estimator, without warnings.

    The class counter ``SpyEstimator.n_fits`` is reset first, so that a test
    reads the number of fits of this very call.

    Args:
        data: Training data.
        **overrides: Constructor parameters, see :func:`make_imputer`.

    Returns:
        The fitted imputer; every plan step model is a ``SpyEstimator``.
    """
    SpyEstimator.n_fits = 0
    imputer = make_imputer(estimator=SpyEstimator(), **overrides)
    return fit_quietly(imputer, data)


# Regroupement d'un index par entité
def by_entity(index: pd.Index) -> Dict[tuple, Set]:
    """Return the set of dates of each entity of an index.

    Args:
        index: ``DatetimeIndex`` or ``MultiIndex`` ``(entity..., date)``.

    Returns:
        ``{entity_tuple: {dates}}``, the empty tuple keying a time series.

    Examples:
        >>> by_entity(pd.DatetimeIndex(['2021-01-31']))
        {(): {Timestamp('2021-01-31 00:00:00')}}
    """
    if not isinstance(index, pd.MultiIndex):
        return {(): set(index)}
    grouped: dict = {}
    for key in index:
        grouped.setdefault(tuple(key[:-1]), set()).add(key[-1])
    return grouped


# Dates renseignées d'une colonne
def filled_dates(frame: pd.DataFrame, column: str) -> Dict[tuple, Set]:
    """Return, per entity, the dates where the column is filled.

    Args:
        frame: Frame to read.
        column: Column to inspect.

    Returns:
        ``{entity_tuple: {dates}}`` of the non-NaN cells.
    """
    return by_entity(frame.index[frame[column].notna().to_numpy()])


# Extraction du bloc d'une entité
def entity_block(series: pd.Series, entity: str) -> pd.Series:
    """Return the slice of one entity, indexed by date alone.

    Args:
        series: Series with an ``(entity, date)`` MultiIndex.
        entity: Value of the entity level.

    Returns:
        The entity's slice, entity level dropped.
    """
    return series.xs(entity, level=0, drop_level=True)


# Contexte d'ajustement d'une (étape, variable), lu par voie publique
@dataclass(frozen=True)
class CapturedFit:
    """Fitting context of one (stage, variable), read through public paths.

    Attributes:
        training: The mutualized :class:`TrainingSet` handed back by
            ``TrainingSetBuilder.build`` — raw target, ``row_origin``,
            ``row_frequency``, ``blocks``, ``has_frequency_level``.
        y_train: The scaled target the estimator was fitted on, read on the
            plan step model (``step.model.fit_y_``).
    """

    training: TrainingSet
    y_train: pd.Series


# Libellé d'une étape de la progression
def stage_label(stage: Union[str, Dict[tuple, str]]) -> str:
    """Return the label of a stage, as ``ImputationStep.pred_freq_label`` names it.

    A time-series stage is a frequency string; a panel stage binds entities
    to frequencies and is labelled by the frequency they all share, or by a
    composite ``entity=freq`` label when they diverge.

    Args:
        stage: Entry of ``frequency_progression_``, or the ``stage_freq``
            argument of ``TrainingSetBuilder.build``.

    Returns:
        The stage label.

    Examples:
        >>> stage_label('M')
        'M'
        >>> stage_label({('FR',): 'Q', ('DE',): 'Q'})
        'Q'
    """
    if not isinstance(stage, dict):
        return str(stage)
    shared = {normalize_frequency(freq, return_format='base') for freq in stage.values()}
    if len(shared) == 1:
        return shared.pop()
    return '+'.join(sorted(f"{entity}={freq}" for entity, freq in stage.items()))


# Capture des contextes d'ajustement de chaque variable
def capture_fits(fit: Callable[[], HighFrequencyImputer]) -> tuple:
    """Run a fit, keeping the fitting context of each (stage, variable).

    The training set is caught on the public ``TrainingSetBuilder.build``:
    the imputer calls it twice per (stage, variable) — a covariate-free probe,
    then the complete set PHASE 5c fits — and the ordering of PHASE 5b goes
    through the same composition, so the LAST call of a (stage, variable) is
    the fitted one. The scaled target is read on the plan step model, which
    must therefore record it (``SpyEstimator``, ``RecordingEstimator``).

    Args:
        fit: Callable building and fitting the imputer.

    Returns:
        Pair ``(imputer, fits)``, ``fits`` mapping ``(stage label, column)``
        to a :class:`CapturedFit`, for every plan step whose model was fitted.
    """
    trainings = {}
    original = TrainingSetBuilder.build

    def _spy(builder, **kwargs):
        training = original(builder, **kwargs)
        trainings[(stage_label(kwargs['stage_freq']), kwargs['column'])] = training
        return training

    with patch.object(TrainingSetBuilder, 'build', _spy):
        imputer = fit()

    fits = {}
    for step in imputer.imputation_plan_:
        key = (step.pred_freq_label, step.var_name)
        if key in trainings and hasattr(step.model, 'fit_y_'):
            fits[key] = CapturedFit(trainings[key], step.model.fit_y_)
    return imputer, fits


# Capture des contextes d'ajustement sous l'estimateur espion
def capture_variable_fits(data: pd.DataFrame, **overrides) -> tuple:
    """Fit a spy imputer, keeping the fitting context of each (stage, variable).

    Args:
        data: Training data.
        **overrides: Constructor parameters, see :func:`fit_with_spy`.

    Returns:
        Pair ``(imputer, fits)``, see :func:`capture_fits`.
    """
    return capture_fits(lambda: fit_with_spy(data, **overrides))


# Groupes imputables d'une étape, lus sur le plan
def plan_groups(imputer: HighFrequencyImputer, stage: str) -> Dict[tuple, object]:
    """Return the imputable groups of one stage, as the plan froze them.

    Args:
        imputer: Fitted imputer.
        stage: Stage label.

    Returns:
        ``{(column, source_frequency): entities}``, ``entities`` being None on
        a time series.
    """
    return {
        (step.var_name, step.source_frequency): step.entities
        for step in imputer.imputation_plan_
        if step.pred_freq_label == stage
    }


# Niveau cible d'une sortie, empilée ou non
def target_level(imputer: HighFrequencyImputer, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the target-frequency level of an output-shaped frame.

    Args:
        imputer: Fitted imputer, whose last progression stage is the target.
        frame: Output of ``transform`` or ``imputation_provenance_`` after it,
            stacked on the frequency level or flat.

    Returns:
        The frame at the target level, frequency level dropped.
    """
    if not has_frequency_level_in_output(frame):
        return frame
    label = stage_label(imputer.frequency_progression_[-1])
    return frame.xs(label, level='frequency')


# Rejeu public du plan : valeurs et provenances au niveau cible
def replay(imputer: HighFrequencyImputer, data: pd.DataFrame) -> tuple:
    """Transform data with a fitted imputer, at the target level.

    The public counterpart of the imputer's internal mirror: the output
    lays the imputations over the input, imputation first (D37).

    Args:
        imputer: Fitted imputer.
        data: Data to transform, usually the fitted data.

    Returns:
        Pair ``(values, provenance)`` of frames at the target level.
    """
    values = target_level(imputer, transform_quietly(imputer, data))
    provenance = target_level(imputer, imputer.imputation_provenance_)
    return values, provenance


# Cellules imputées d'une colonne
def imputed_cells(values: pd.DataFrame, provenance: pd.DataFrame, column: str) -> pd.Series:
    """Return the cells of a column the imputer wrote, out of a replay.

    A cell is imputed when its provenance is set and differs from
    ``ORIGINAL``.

    Args:
        values: Output values, see :func:`replay`.
        provenance: Matching provenance, see :func:`replay`.
        column: Column to read.

    Returns:
        The imputed cells, possibly empty.
    """
    marks = provenance[column]
    written = marks.notna() & (marks.astype(str) != 'original')
    return values.loc[written.reindex(values.index, fill_value=False), column]


# Lecture des couples (étape, variable) du plan
def plan_pairs(imputer: HighFrequencyImputer) -> List[Tuple[str, str]]:
    """Return the ordered (stage label, variable) pairs of the plan.

    Args:
        imputer: Fitted imputer.

    Returns:
        One pair per plan step, in plan order.
    """
    return [(step.pred_freq_label, step.var_name) for step in imputer.imputation_plan_.steps]


# Présence du niveau de fréquence de la sortie empilée
def has_frequency_level_in_output(frame: pd.DataFrame) -> bool:
    """Tell whether a frame carries the ``'frequency'`` level of the stacked output.

    Args:
        frame: Frame to inspect.

    Returns:
        True if its index is a MultiIndex naming a ``'frequency'`` level.
    """
    return (
        isinstance(frame.index, pd.MultiIndex)
        and 'frequency' in (frame.index.names or [])
    )
