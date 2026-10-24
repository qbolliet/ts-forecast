"""Scenario matrix and runner of the ``HighFrequencyImputer`` golden master.

Defines the frozen datasets (``PANEL-X``, its three projections and the two
realistic datasets of notebook 3, each split into a fit part and a posterior
part), the matrix of configurations pruned to its equivalence classes
(:func:`build_cases`), the run of one case (:func:`run_case`: training pass,
inversion, prediction pass on the posterior data, inversion again) and the
canonical form of every recorded object (:func:`canonicalize`), made of
pandas, numpy and builtin objects only.
"""
# Modules de base
import dataclasses
import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

# Composant figé
from tsforecast.imputation import HighFrequencyImputer

# Jeux de données partagés
from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_mixed_frequency_panel,
    build_mixed_frequency_timeseries,
    build_panel_reference,
)

# Paramètres communs à tous les cas : grille mensuelle, sortie empilée par
# défaut, restitution des valeurs d'origine pour exercer toute l'inversion
COMMON_PARAMS: Dict[str, Any] = {
    'target_frequency': 'M',
    'restore_original_values': True,
}

# Graine du découpage de la validation croisée de l'ordre 'cv'
CV_SEED = 0
CV_SPLITS = 3


# =============================================================================
# Jeux de données
# =============================================================================
@dataclasses.dataclass(frozen=True)
class GoldenDataset:
    """Frozen dataset of the golden master and how a case consumes it.

    Attributes:
        name: Identifier used in the case ids.
        build: Zero-argument builder returning the full dataset.
        cutoff: First date of the posterior part: the training pass runs on
            the dates strictly before it, the prediction pass on the others.
        target: Column taken out of ``X`` as ``y`` by the cases fitted with
            a target.
    """

    name: str
    build: Callable[[], pd.DataFrame]
    cutoff: str
    target: str


# Projection TS de PANEL-X, identique à la fixture "reference_timeseries"
def _reference_timeseries() -> pd.DataFrame:
    """Return the ``TS`` projection of ``PANEL-X`` (entity ``FR``, 4 columns)."""
    frame = build_panel_reference().loc['FR', ['m1', 'q1', 'a1', 'a2']].copy()
    # Restauration de la fréquence d'index, perdue au découpage du MultiIndex
    frame.index.freq = frame.index.inferred_freq
    return frame


# Projections en colonnes de PANEL-X, identiques aux fixtures du même nom
def _reference_panel() -> pd.DataFrame:
    """Return the ``PANEL`` projection of ``PANEL-X``."""
    return build_panel_reference()[['m1', 'q1', 'a1', 'a2', 'climat_affaires']].copy()


def _reference_panel_multifrequency() -> pd.DataFrame:
    """Return the ``PANEL-F`` projection of ``PANEL-X``."""
    return build_panel_reference()[['m1', 'q1', 'v']].copy()


# Jeux figés : PANEL-X, ses trois projections, les deux jeux réalistes.
# Coupure au 1er janvier 2023 : deux années d'ajustement pour PANEL-X (trois
# ancres annuelles par entité au moins), le reste en données postérieures.
DATASETS: Dict[str, GoldenDataset] = {
    dataset.name: dataset
    for dataset in (
        GoldenDataset('panel_x', build_panel_reference, '2023-01-01', 'q1'),
        GoldenDataset('ts', _reference_timeseries, '2023-01-01', 'q1'),
        GoldenDataset('panel', _reference_panel, '2023-01-01', 'q1'),
        GoldenDataset('panel_f', _reference_panel_multifrequency, '2023-01-01', 'q1'),
        GoldenDataset(
            'irregular_ts',
            lambda: build_mixed_frequency_timeseries(annual_start_date='2015-01-01'),
            '2023-01-01',
            'pib_trimestriel',
        ),
        GoldenDataset(
            'heterogeneous_panel',
            lambda: build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES),
            '2023-01-01',
            'pib_trimestriel',
        ),
    )
}


def split_dataset(dataset: GoldenDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into its fit part and its posterior part.

    Args:
        dataset: Dataset of :data:`DATASETS`.

    Returns:
        ``(fit_part, posterior_part)``: the rows dated strictly before
        ``dataset.cutoff``, then the others, in their original order.

    Examples:
        >>> fit_part, posterior = split_dataset(DATASETS['ts'])
        >>> len(fit_part), len(posterior)
        (24, 12)
    """
    data = dataset.build()
    # Lecture des dates sur le dernier niveau : index simple ou panel
    dates = data.index.get_level_values(-1)
    before = np.asarray(dates < pd.Timestamp(dataset.cutoff))
    return data.loc[before].copy(), data.loc[~before].copy()


def has_unobserved_pairs(data: pd.DataFrame) -> bool:
    """Tell whether a panel holds an entity that never observes a column.

    ``impute_unobserved_entities`` only acts on such pairs: without one, its
    two values give the same result.

    Args:
        data: Dataset, time series or panel.

    Returns:
        True on a panel where at least one (entity, column) pair is entirely
        NaN, False otherwise (always False on a time series).

    Examples:
        >>> fit_part, _ = split_dataset(DATASETS['panel_x'])
        >>> has_unobserved_pairs(fit_part)
        True
        >>> has_unobserved_pairs(split_dataset(DATASETS['ts'])[0])
        False
    """
    if data.index.nlevels == 1:
        return False
    # Décompte des observations par (entité, colonne)
    counts = data.notna().groupby(level=list(range(data.index.nlevels - 1))).sum()
    return bool((counts == 0).to_numpy().any())


# =============================================================================
# Matrice des configurations
# =============================================================================
# Classes d'équivalence des deux axes et de l'ordre d'imputation :
# - 'covariates_only' ne change aucune valeur finale hors covariate_strategy='model'
#   (les rangs de matérialisation qui lui donnent un effet sont hors d'atteinte) ;
# - fit_predict_order est inerte hors covariate_strategy='model'.
# Restent 2 + 2 modalités hors 'model' et 3 x 2 sous 'model' : 10 classes.
AXIS_CLASSES: Tuple[Tuple[str, Union[bool, str], str], ...] = (
    ('tolerate_nan', False, 'frequency'),
    ('tolerate_nan', True, 'frequency'),
    ('interpolate', False, 'frequency'),
    ('interpolate', True, 'frequency'),
    ('model', False, 'frequency'),
    ('model', False, 'cv'),
    ('model', 'covariates_only', 'frequency'),
    ('model', 'covariates_only', 'cv'),
    ('model', True, 'frequency'),
    ('model', True, 'cv'),
)

# Contrainte d'agrégation : effective sous toutes les modalités (recalage)
AGGREGATION_CONSTRAINTS: Tuple[Optional[str], ...] = ('sum', None)

# Estimateurs : régression linéaire précédée d'une imputation par la moyenne
# (tolère les trous des covariables : les étapes s'ajustent réellement), ou
# régression linéaire nue (échoue sur les trous : chemin de repli par
# interpolation)
NAN_TOLERANT_ESTIMATOR = 'imputed_linear'
BARE_ESTIMATOR = 'linear'

# Jeux des groupes secondaires : jeux à couple jamais observé (PANEL n'en est
# qu'une projection de colonnes), jeux ajustés avec y (une série régulière, une
# série irrégulière, le panel de référence, le panel réaliste), jeux de
# l'estimateur nu (sur PANEL-X, toutes les étapes basculent en repli)
UNOBSERVED_DATASETS: Tuple[str, ...] = ('panel_x', 'heterogeneous_panel')
TARGET_DATASETS: Tuple[str, ...] = ('ts', 'irregular_ts', 'panel_x', 'heterogeneous_panel')
BARE_ESTIMATOR_DATASETS: Tuple[str, ...] = ('ts', 'panel_x')


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    """One cell of the golden master matrix.

    Attributes:
        dataset: Key of :data:`DATASETS`.
        covariate_strategy: Axis 1.
        impute_intermediate_frequencies: Axis 2 (``False``,
            ``'covariates_only'`` or ``True``).
        fit_predict_order: ``'frequency'`` or ``'cv'``.
        aggregation_constraint: ``'sum'`` or ``None``.
        impute_unobserved_entities: Whether unobserved pairs are imputed.
        with_target: Whether the case is fitted with ``y`` (the dataset's
            ``target`` column), running the training and prediction passes.
        estimator: :data:`NAN_TOLERANT_ESTIMATOR` or :data:`BARE_ESTIMATOR`.
    """

    dataset: str
    covariate_strategy: str
    impute_intermediate_frequencies: Union[bool, str]
    fit_predict_order: str
    aggregation_constraint: Optional[str]
    impute_unobserved_entities: bool
    with_target: bool
    estimator: str = NAN_TOLERANT_ESTIMATOR

    @property
    def case_id(self) -> str:
        """Readable identifier, also the stem of the case's golden file.

        Examples:
            >>> GoldenCase('ts', 'model', 'covariates_only', 'cv', None, False, True).case_id
            'ts__model__covariates_only__cv__none__observed__xy__imputed_linear'
        """
        modality = str(self.impute_intermediate_frequencies).lower()
        constraint = str(self.aggregation_constraint).lower()
        unobserved = 'unobserved' if self.impute_unobserved_entities else 'observed'
        target = 'xy' if self.with_target else 'x'
        return '__'.join((
            self.dataset, self.covariate_strategy, modality,
            self.fit_predict_order, constraint, unobserved, target, self.estimator,
        ))

    def imputer_params(self) -> Dict[str, Any]:
        """Return the JSON-serializable parameters specific to the case.

        Returns:
            The five parameters of the matrix, under their ``__init__`` names.
        """
        return {
            'covariate_strategy': self.covariate_strategy,
            'impute_intermediate_frequencies': self.impute_intermediate_frequencies,
            'fit_predict_order': self.fit_predict_order,
            'aggregation_constraint': self.aggregation_constraint,
            'impute_unobserved_entities': self.impute_unobserved_entities,
        }


def build_cases() -> List[GoldenCase]:
    """Build the matrix of the golden master, pruned to distinct combinations.

    Four groups, each running the 10 axis classes of :data:`AXIS_CLASSES`:

    - main: every dataset x both aggregation constraints, ``X`` alone,
      NaN-tolerant estimator;
    - unobserved pairs: ``impute_unobserved_entities=True`` (constraint
      ``'sum'``) on :data:`UNOBSERVED_DATASETS`, whose fit part holds an
      entity never observing a column — elsewhere the parameter acts on
      nothing;
    - target: fitted with ``y`` (constraint ``'sum'``) on
      :data:`TARGET_DATASETS`, running the training and prediction passes;
    - bare estimator: ``LinearRegression`` without imputation (constraint
      ``'sum'``) on :data:`BARE_ESTIMATOR_DATASETS`, whose failed fits run the
      interpolation fallback.

    Returns:
        The cases, in a stable order.

    Raises:
        ValueError: If a dataset of :data:`UNOBSERVED_DATASETS` has no
            unobserved pair in its fit part.

    Examples:
        >>> cases = build_cases()
        >>> len(cases), len({case.case_id for case in cases})
        (200, 200)
    """
    cases: List[GoldenCase] = []
    for name in DATASETS:
        for strategy, modality, order in AXIS_CLASSES:
            for constraint in AGGREGATION_CONSTRAINTS:
                cases.append(GoldenCase(name, strategy, modality, order, constraint, False, False))
    for name in UNOBSERVED_DATASETS:
        if not has_unobserved_pairs(split_dataset(DATASETS[name])[0]):
            raise ValueError(f'{name}: no unobserved (entity, column) pair to impute')
        for strategy, modality, order in AXIS_CLASSES:
            cases.append(GoldenCase(name, strategy, modality, order, 'sum', True, False))
    for name in TARGET_DATASETS:
        for strategy, modality, order in AXIS_CLASSES:
            cases.append(GoldenCase(name, strategy, modality, order, 'sum', False, True))
    for name in BARE_ESTIMATOR_DATASETS:
        for strategy, modality, order in AXIS_CLASSES:
            cases.append(GoldenCase(
                name, strategy, modality, order, 'sum', False, False, BARE_ESTIMATOR
            ))
    return cases


def make_estimator(estimator: str) -> BaseEstimator:
    """Return a fresh deterministic estimator.

    Args:
        estimator: :data:`NAN_TOLERANT_ESTIMATOR` or :data:`BARE_ESTIMATOR`.

    Returns:
        ``make_pipeline(SimpleImputer(), LinearRegression())`` or
        ``LinearRegression()``.

    Raises:
        ValueError: If the name is unknown.

    Examples:
        >>> type(make_estimator('linear')).__name__
        'LinearRegression'
        >>> type(make_estimator('imputed_linear')).__name__
        'Pipeline'
    """
    if estimator == NAN_TOLERANT_ESTIMATOR:
        return make_pipeline(SimpleImputer(), LinearRegression())
    if estimator == BARE_ESTIMATOR:
        return LinearRegression()
    raise ValueError(f'Unknown estimator {estimator!r}')


def make_imputer(case: GoldenCase) -> HighFrequencyImputer:
    """Build the imputer of a case.

    Args:
        case: Cell of the matrix.

    Returns:
        Unfitted imputer: :data:`COMMON_PARAMS`, the case parameters, the
        case estimator and a shuffled ``KFold`` of fixed seed.
    """
    return HighFrequencyImputer(
        estimator=make_estimator(case.estimator),
        cv=KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED),
        **COMMON_PARAMS,
        **case.imputer_params(),
    )


# =============================================================================
# Forme canonique des objets enregistrés
# =============================================================================
def canonicalize(value: Any) -> Any:
    """Convert a recorded object into pandas, numpy and builtin objects only.

    The golden files must load whatever the module layout of ``tsforecast``:
    no class of the package may be pickled. Enum members become their value,
    dataclasses and other objects (estimators, splitters) become a dict
    holding their class name under ``'__class__'`` and their public
    attributes; mappings become plain dicts, keys included.

    Args:
        value: Any object read on the imputer or returned by it.

    Returns:
        The canonical form; pandas and numpy objects are copied.

    Raises:
        TypeError: If the object is of a kind the golden master cannot record.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> canonicalize({('FR', 'q1'): (1, 'Q')})
        {('FR', 'q1'): (1, 'Q')}
        >>> canonicalize(LinearRegression())['__class__']
        'LinearRegression'
    """
    # Membres d'énumération avant les chaînes : ProvenanceType hérite de str
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta, pd.Period, type(pd.NaT))):
        return value
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
        return value.copy()
    if isinstance(value, Mapping):
        return {canonicalize(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(canonicalize(item) for item in value)
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return frozenset(canonicalize(item) for item in value)
    # Classes de données : champs déclarés, dans leur ordre
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return {'__class__': type(value).__name__, **fields}
    # Estimateurs, découpages : attributs publics, paramètres et état ajusté
    if hasattr(value, '__dict__') and not callable(value) or isinstance(value, BaseEstimator):
        attributes = {
            name: canonicalize(item)
            for name, item in sorted(vars(value).items())
            if not name.startswith('_')
        }
        return {'__class__': type(value).__name__, **attributes}
    raise TypeError(f'Cannot record an object of type {type(value).__name__}: {value!r}')


# =============================================================================
# Exécution d'un cas
# =============================================================================
def recorded_call(call: Callable[[], Any]) -> Dict[str, Any]:
    """Run a call and record its result, its exception and its warnings.

    Args:
        call: Zero-argument callable.

    Returns:
        ``{'result': canonical result or None, 'error': (exception class
        name, message) or None, 'warnings': [(category name, message), ...]}``,
        the warnings in emission order, repeats included.

    Examples:
        >>> import warnings
        >>> recorded_call(lambda: warnings.warn('w') or 1)['warnings']
        [('UserWarning', 'w')]
        >>> recorded_call(lambda: 1 / 0)['error']
        ('ZeroDivisionError', 'division by zero')
    """
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 — l'exception fait partie du comportement figé
            error = (type(exc).__name__, str(exc))
    return {
        'result': canonicalize(result),
        'raw_result': result,
        'error': error,
        'warnings': [(item.category.__name__, str(item.message)) for item in caught],
    }


def public_attribute_names(imputer: HighFrequencyImputer) -> List[str]:
    """List the public fitted attributes present on an imputer.

    Args:
        imputer: Imputer, fitted or not.

    Returns:
        Sorted names ending with ``_`` and not starting with ``_`` that can be
        read on the instance (properties included).
    """
    names = []
    for name in dir(imputer):
        if not name.endswith('_') or name.startswith('_'):
            continue
        try:
            getattr(imputer, name)
        except AttributeError:
            continue
        names.append(name)
    return sorted(names)


def public_state(imputer: HighFrequencyImputer) -> Dict[str, Any]:
    """Record the public fitted state of an imputer.

    Args:
        imputer: Fitted imputer.

    Returns:
        ``{'attributes': {name: canonical value}, 'plan_diagnostic':
        imputation_plan_.to_diagnostic_frame()}``; the provenance matrix is
        the attribute ``imputation_provenance_``.
    """
    attributes = {
        name: canonicalize(getattr(imputer, name))
        for name in public_attribute_names(imputer)
    }
    return {
        'attributes': attributes,
        'plan_diagnostic': imputer.imputation_plan_.to_diagnostic_frame(),
    }


def _strip_raw(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the raw result of a recorded call, kept for chaining only."""
    return {key: value for key, value in record.items() if key != 'raw_result'}


def run_case(case: GoldenCase) -> Dict[str, Any]:
    """Run one case of the golden master and record everything it produces.

    Four calls, each recorded by :func:`recorded_call`: ``fit_transform`` on
    the fit part (with ``y`` for the cases fitted with a target: the training
    pass), ``inverse_transform`` of its output, ``transform`` of the posterior
    part (``X`` alone: the prediction pass when fitted with ``y``), and
    ``inverse_transform`` of that output. The public state is recorded after
    the first and the third call. A call that raises ends the record.

    Args:
        case: Cell of the matrix.

    Returns:
        Dict of canonical objects, keys ``fit_transform``,
        ``state_after_fit_transform``, ``inverse_fit_transform``,
        ``transform``, ``state_after_transform``, ``inverse_transform``
        (those after a raising call are absent).
    """
    dataset = DATASETS[case.dataset]
    fit_part, posterior = split_dataset(dataset)
    imputer = make_imputer(case)

    # Passe d'entraînement : X seul, ou X et y extraits du même jeu
    if case.with_target:
        X_fit, y_fit = fit_part.drop(columns=dataset.target), fit_part[dataset.target]
        X_posterior = posterior.drop(columns=dataset.target)
        fitted = recorded_call(lambda: imputer.fit_transform(X_fit, y_fit))
    else:
        X_posterior = posterior
        fitted = recorded_call(lambda: imputer.fit_transform(fit_part))

    record: Dict[str, Any] = {'fit_transform': _strip_raw(fitted)}
    if fitted['error'] is not None:
        return record
    record['state_after_fit_transform'] = public_state(imputer)

    # Inversion de la passe d'entraînement, y compris de y
    output = fitted['raw_result']
    inverted = recorded_call(
        lambda: imputer.inverse_transform(*output) if case.with_target
        else imputer.inverse_transform(output)
    )
    record['inverse_fit_transform'] = _strip_raw(inverted)

    # Passe de prédiction sur les données postérieures
    transformed = recorded_call(lambda: imputer.transform(X_posterior))
    record['transform'] = _strip_raw(transformed)
    if transformed['error'] is not None:
        return record
    record['state_after_transform'] = public_state(imputer)

    # Inversion de la passe de prédiction
    output_posterior = transformed['raw_result']
    record['inverse_transform'] = _strip_raw(
        recorded_call(lambda: imputer.inverse_transform(output_posterior))
    )
    return record
