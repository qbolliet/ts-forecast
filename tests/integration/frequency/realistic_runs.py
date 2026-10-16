"""Scenario matrix of ``HighFrequencyImputer`` on the realistic datasets of notebook 3.

Shared by ``test_hfi_on_realistic_dataset.py`` and ``test_tracking.py``
(through the session-scoped ``realistic_run`` fixture of this folder's
``conftest.py``), so that each cell of the matrix is fitted once per session:
dataset {``'ts'``, ``'panel'``} x ``covariate_strategy`` {``'interpolate'``,
``'model'``, ``'tolerate_nan'``} x ``impute_intermediate_frequencies``
{``False``, ``'covariates_only'``, ``True``}.
"""
# Modules de base
import warnings
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Composant ajusté
from tsforecast.frequency import HighFrequencyImputer

# Estimateur de chaque stratégie : 'tolerate_nan' exige un estimateur tolérant
# aux NaN (même choix que les tests unitaires réalistes)
STRATEGY_ESTIMATORS: Dict[str, Callable[[], object]] = {
    'interpolate': LinearRegression,
    'model': LinearRegression,
    'tolerate_nan': lambda: HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=2),
}

# Modalités de impute_intermediate_frequencies, jamais testées par vérité booléenne
MODALITIES: List[Union[bool, str]] = [False, 'covariates_only', True]

# Jeux réalistes : nom de la fixture de session (tests/support/fixtures.py) qui les construit
DATASET_FIXTURES: Dict[str, str] = {
    'ts': '_irregular_index_timeseries_session',
    'panel': '_heterogeneous_coverage_panel_session',
}


def matrix_case(dataset: str, strategy: str, modality: Union[bool, str]):
    """Build one ``pytest.param`` of the matrix, ``slow`` for the panel.

    Args:
        dataset: ``'ts'`` or ``'panel'``.
        strategy: Value of ``covariate_strategy``.
        modality: Value of ``impute_intermediate_frequencies``.

    Returns:
        ``pytest.param(dataset, strategy, modality)`` with a readable id such as
        ``ts-model-covariates_only``; the panel cells (2.5-6 s per fit) are
        marked ``slow``.
    """
    marks = [pytest.mark.slow] if dataset == 'panel' else []
    return pytest.param(
        dataset, strategy, modality, marks=marks, id=f'{dataset}-{strategy}-{modality}'
    )


# Produit complet : 2 jeux x 3 stratégies x 3 modalités = 18 ajustements
CASES = [
    matrix_case(dataset, strategy, modality)
    for dataset in DATASET_FIXTURES
    for strategy in STRATEGY_ESTIMATORS
    for modality in MODALITIES
]


def caught_messages(call: Callable[[], object]) -> Tuple[object, List[str]]:
    """Run a call and return its result with every warning message it emitted.

    Args:
        call: Zero-argument callable.

    Returns:
        ``(result, messages)``, the messages in emission order.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return result, [str(item.message) for item in caught]


def build_record(data: pd.DataFrame, strategy: str, modality: Union[bool, str]) -> dict:
    """Fit, transform and invert one cell of the matrix.

    The imputer keeps its defaults (``target_frequency='M'``, stacked output
    ``keep_lower_frequencies=True``, ``aggregation_constraint='sum'``,
    ``imputation_scope='strict'``), plus ``restore_original_values=True``,
    which only affects ``inverse_transform``. ``fit`` and ``transform`` are
    called separately - ``fit_transform`` is ``fit(X).transform(X)`` (I1) -
    so that the warnings of each pass are recorded apart.

    Args:
        data: Realistic dataset, never modified.
        strategy: Value of ``covariate_strategy``.
        modality: Value of ``impute_intermediate_frequencies``.

    Returns:
        Record holding the input ``data``, the fitted ``imputer``, the stacked
        ``output`` of the transform, its ``provenance``, the ``inverse`` of the
        output, and the warning messages of the fit (``fit_messages``) and of
        the transform (``transform_messages``).
    """
    imputer = HighFrequencyImputer(
        target_frequency='M',
        estimator=STRATEGY_ESTIMATORS[strategy](),
        covariate_strategy=strategy,
        impute_intermediate_frequencies=modality,
        restore_original_values=True,
    )
    _, fit_messages = caught_messages(lambda: imputer.fit(data))
    output, transform_messages = caught_messages(lambda: imputer.transform(data))
    provenance = imputer.imputation_provenance_.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        inverse = imputer.inverse_transform(output)
    return {
        'data': data,
        'imputer': imputer,
        'output': output,
        'provenance': provenance,
        'inverse': inverse,
        'fit_messages': fit_messages,
        'transform_messages': transform_messages,
    }
