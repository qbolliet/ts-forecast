"""Catalogue of public calls shared by the input-contract tests (not a test module).

Each :class:`Case` builds its inputs from ``tests/support/datasets.py`` and runs
one public component end to end (``fit`` / ``transform`` / ``inverse_transform``,
``split``, or a function), registering through a ``guard`` callback every object
it hands to the package — including the output it passes back to
``inverse_transform``. :func:`snapshot` / :func:`assert_unchanged` freeze and
compare those objects; :func:`as_read_only` rebuilds a float frame on a
read-only NumPy buffer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_mixed_frequency_panel,
    build_mixed_frequency_timeseries,
    build_panel_reference,
    build_staggered_panel,
)
from tsforecast.crossvals import (
    InSampleSplit,
    OutOfSampleSplit,
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
    TSInSampleSplit,
    TSOutOfSampleSplit,
)
from tsforecast.delays import (
    MaskTransformer,
    PublicationDelayTransformer,
    ShiftTransformer,
    calculate_applicable_delay,
    compare_and_detect_delays,
    create_delay_transformer_factory,
)
from tsforecast.utils.frequency import FrequencyAligner, is_regular, regularize
from tsforecast.imputation import HighFrequencyImputer
from tsforecast.panel import PanelwiseTransformer
from tsforecast.tracking import split_summary
from tsforecast.utils.frequency import FrequencyConverter, detect_dataset_frequency, detect_frequency
from tsforecast.utils.position import convert_position
from tsforecast.utils.validation import validate_temporal_data
from tsforecast.xy import DropNaNTarget, XYPipeline

# Fonction d'enregistrement : reçoit un objet transmis au paquet et le renvoie
Guard = Callable[[Any], Any]

# Cible des cas (X, y) : mensuelle, observée sur toute la grille sauf la dernière ligne
TARGET = 'taux_chomage'

# Date de prédiction des transformateurs de délais
PREDICTION_DATE = '2024-08-15'

# Délais en jours depuis le début de la période
DELAYS = {'inflation_ipc': 20.0, 'taux_chomage': 50.0, 'pib_trimestriel': 75.0}


@dataclass(frozen=True)
class Case:
    """One public call: ``build()`` gives the inputs, ``run(inputs, guard)`` calls the package.

    Attributes:
        build: Zero-argument builder of the inputs, a ``{name: object}`` mapping.
        run: Callable receiving the inputs and the guard, returning the outputs
            (``{name: object}``) to compare between two runs.
        slow: Whether the case takes more than about a second.
    """

    build: Callable[[], Dict[str, Any]]
    run: Callable[[Dict[str, Any], Guard], Dict[str, Any]]
    slow: bool = False


# =============================================================================
# Jeux d'entrée
# =============================================================================
def _series() -> Dict[str, Any]:
    """Monthly mixed-frequency series (regular grid)."""
    return {'X': build_mixed_frequency_timeseries()}


def _irregular() -> Dict[str, Any]:
    """Genuinely irregular series of notebook 3."""
    return {'X': build_mixed_frequency_timeseries(annual_start_date='2015-01-01')}


def _panel() -> Dict[str, Any]:
    """Monthly mixed-frequency panel of three countries."""
    return {'X': build_mixed_frequency_panel()}


def _monthly_panel() -> Dict[str, Any]:
    """Monthly columns only of the panel (every column converts with the same method)."""
    return {'X': build_mixed_frequency_panel()[['inflation_ipc', 'taux_chomage']]}


def _xy_series() -> Dict[str, Any]:
    """``(X, y)`` of the irregular series, the target NaN on a few rows."""
    data = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
    return {'X': data.drop(columns=TARGET), 'y': data[TARGET]}


def _reference_ts() -> Dict[str, Any]:
    """``TS`` reference dataset (``m1``, ``q1``, ``a1``, ``a2``), ``y = a2``."""
    data = build_panel_reference().loc['FR', ['m1', 'q1', 'a1', 'a2']].copy()
    data.index.freq = data.index.inferred_freq
    return {'X': data.drop(columns='a2'), 'y': data['a2']}


def _reference_panel() -> Dict[str, Any]:
    """``PANEL-F`` reference dataset (``v`` of entity-specific frequency)."""
    return {'X': build_panel_reference()[['m1', 'q1', 'v']].copy()}


def _staggered() -> Dict[str, Any]:
    """Staggered panel, its target and its entity groups."""
    X = build_staggered_panel()
    return {'X': X, 'y': X['v'] * 2.0, 'groups': pd.Series(X.index.get_level_values(0), index=X.index)}


def _months() -> Dict[str, Any]:
    """Twenty-four month ends with a target."""
    X = pd.DataFrame({'v': np.arange(24.0)}, index=pd.date_range('2023-01-31', periods=24, freq='ME', name='date'))
    return {'X': X, 'y': X['v'] * 3.0}


def _delay_table() -> Dict[str, Any]:
    """Delays detected on the realistic panel (first download)."""
    panel = build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)
    return {'delays': compare_and_detect_delays(panel, None, '2024-10-15')}


def _two_snapshots() -> Dict[str, Any]:
    """Realistic panel and an earlier snapshot without the last observation of each column."""
    new = build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)
    existing = new.copy()
    for column in new.columns:
        last = new[column].dropna().index[-1:]
        existing.loc[last, column] = np.nan
    return {'new': new, 'existing': existing}


# =============================================================================
# Appels
# =============================================================================
def _quiet(function: Callable, *args, **kwargs):
    """Call ``function`` with the warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _transformer_round_trip(make: Callable[[], Any]) -> Callable:
    """``fit``, ``transform`` and ``inverse_transform`` of a fresh transformer on ``X``."""
    def run(inputs, guard):
        transformer = make()
        _quiet(transformer.fit, guard(inputs['X']))
        output = _quiet(transformer.transform, guard(inputs['X']))
        inverse = _quiet(transformer.inverse_transform, guard(output))
        return {'output': output, 'inverse': inverse}
    return run


def _split(make: Callable[[], Any], with_y: bool = True, with_groups: bool = False) -> Callable:
    """Folds of a fresh splitter, and its summary metrics."""
    def run(inputs, guard):
        splitter = make()
        args = [guard(inputs['X'])]
        if with_y:
            args.append(guard(inputs['y']))
        if with_groups:
            args.append(guard(inputs['groups']))
        folds = [(train.tolist(), test.tolist()) for train, test in _quiet(lambda: list(splitter.split(*args)))]
        return {'folds': folds, 'summary': _quiet(split_summary, splitter, guard(inputs['X']))}
    return run


def _imputer_passes(strategy: str, with_y: bool) -> Callable:
    """Training pass, prediction pass and inversion of a ``HighFrequencyImputer``."""
    def run(inputs, guard):
        estimator = HistGradientBoostingRegressor(max_iter=10) if strategy == 'tolerate_nan' else LinearRegression()
        imputer = HighFrequencyImputer(target_frequency='M', estimator=estimator, covariate_strategy=strategy,
                                       restore_original_values=True)
        if with_y:
            X_out, y_out = _quiet(imputer.fit_transform, guard(inputs['X']), guard(inputs['y']))
            outputs = {'X_out': X_out, 'y_out': y_out}
        else:
            outputs = {'output': _quiet(imputer.fit_transform, guard(inputs['X']))}
        prediction = _quiet(imputer.transform, guard(inputs['X']))
        outputs['prediction'] = prediction
        outputs['inverse'] = _quiet(imputer.inverse_transform, guard(prediction))
        return outputs
    return run


def _pipeline(inputs, guard):
    """``XYPipeline`` delays -> imputer -> drop NaN target -> estimator: fit, predict, score."""
    pipeline = XYPipeline([
        ('delays', PublicationDelayTransformer(
            {'inflation_ipc': 20.0}, prediction_date=PREDICTION_DATE, strategy='mask', target_frequency='Q',
            delay_unit='D', reference_point='start', handle_missing_delays='ignore',
        )),
        ('imputer', HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                         keep_lower_frequencies=False)),
        ('drop', DropNaNTarget()),
        ('model', HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=2)),
    ])
    X, y = inputs['X'], inputs['y']
    X_train, y_train, X_test, y_test = X.iloc[:-6], y.iloc[:-6], X.iloc[-6:], y.iloc[-6:]
    _quiet(pipeline.fit, guard(X_train), guard(y_train))
    predictions = _quiet(pipeline.predict, guard(X_test))
    score = _quiet(pipeline.score, guard(X_test.iloc[:-1]), guard(y_test.iloc[:-1]))
    return {'predictions': np.asarray(predictions), 'score': score}


def _drop_nan_target(inputs, guard):
    """``DropNaNTarget`` on ``(X, y)`` whose target has missing values."""
    X_out, y_out = _quiet(DropNaNTarget().fit_transform, guard(inputs['X']), guard(inputs['y']))
    return {'X_out': X_out, 'y_out': y_out}


def _panelwise_scaler(inputs, guard):
    """``PanelwiseTransformer`` around ``StandardScaler``: fit, transform, inverse."""
    transformer = PanelwiseTransformer(StandardScaler())
    output = _quiet(transformer.fit_transform, guard(inputs['X']))
    return {'output': output, 'inverse': _quiet(transformer.inverse_transform, guard(output))}


def _panelwise_delays(inputs, guard):
    """Per-entity delays applied by ``PanelwiseTransformer`` through the factory."""
    delays = pd.DataFrame(
        {'delay': [20.0, 50.0, 45.0], 'unit': 'D', 'reference_point': 'start', 'frequency': 'Q'},
        index=pd.MultiIndex.from_tuples(
            [('France', 'inflation_ipc'), ('Allemagne', 'taux_chomage'), ('Italie', 'inflation_ipc')],
            names=['country', 'column'],
        ),
    )
    factory = create_delay_transformer_factory(guard(delays), strategy='mask', prediction_date=PREDICTION_DATE)
    transformer = PanelwiseTransformer(transformer=factory, time_col=None, panel_cols=None)
    return {'output': _quiet(transformer.fit_transform, guard(inputs['X']))}


def _functions(inputs, guard):
    """Validation, detection, conversion, position and regularity functions on ``X``."""
    X = inputs['X']
    converter = FrequencyConverter()
    return {
        'validated': _quiet(validate_temporal_data, guard(X)),
        'columns_path': _quiet(validate_temporal_data, guard(X.reset_index()), time_col='date'),
        'frequency': _quiet(detect_frequency, guard(X), return_format='full'),
        'dataset_frequency': _quiet(detect_dataset_frequency, guard(X), return_format='full'),
        'quarterly': _quiet(converter.aggregate_to_lower_frequency, guard(X['inflation_ipc']), 'QS', method='sum'),
        'monthly': _quiet(converter.interpolate_to_higher_frequency, guard(X['pib_trimestriel'].dropna()), 'MS'),
        'weekly': _quiet(converter.convert_frequency, guard(X[['inflation_ipc']]), 'W', method='linear'),
        'position': _quiet(convert_position, guard(X), 'start', 'end', freq='M'),
        'aligned': _quiet(FrequencyAligner().convert_to_target, guard(X), ['pib_trimestriel'], 'M'),
    }


def _regularity(inputs, guard):
    """``is_regular`` and ``regularize`` on the irregular series."""
    return {'regular': _quiet(is_regular, guard(inputs['X'])), 'regularized': _quiet(regularize, guard(inputs['X']))}


def _panel_conversion(inputs, guard):
    """Panel conversion and detection."""
    return {
        'quarterly': _quiet(FrequencyConverter().convert_frequency, guard(inputs['X']), 'QS', method='mean'),
        'frequencies': _quiet(detect_dataset_frequency, guard(inputs['X']), return_format='full'),
    }


def _detection(inputs, guard):
    """Delay detection from two snapshots and from a single download."""
    return {
        'two': _quiet(compare_and_detect_delays, guard(inputs['new']), guard(inputs['existing']), '2024-10-15'),
        'single': _quiet(compare_and_detect_delays, guard(inputs['new']), None, '2024-10-15', delay_unit='s'),
    }


def _applicable(inputs, guard):
    """Applicable delays from a detected delay table."""
    return {'applicable': _quiet(calculate_applicable_delay, guard(inputs['delays']), 'end', 'Q', unit='D',
                                 aggregate_by_panel=True)}


# Catalogue : nom lisible -> cas
CASES: Dict[str, Case] = {
    'validation-detection-conversion': Case(_series, _functions),
    'regularity': Case(_irregular, _regularity),
    'panel-conversion': Case(_monthly_panel, _panel_conversion),
    'delay-detection': Case(_two_snapshots, _detection),
    'applicable-delay': Case(_delay_table, _applicable),
    'shift-transformer': Case(_series, _transformer_round_trip(lambda: ShiftTransformer(2, 'M'))),
    'mask-transformer': Case(_series, _transformer_round_trip(lambda: MaskTransformer(1, 'Q', how='first'))),
    'publication-delay-shift': Case(_series, _transformer_round_trip(lambda: PublicationDelayTransformer(
        DELAYS, prediction_date=PREDICTION_DATE, strategy='shift', delay_unit='D', reference_point='start',
        handle_missing_delays='ignore',
    ))),
    'publication-delay-mask': Case(_irregular, _transformer_round_trip(lambda: PublicationDelayTransformer(
        DELAYS, prediction_date=PREDICTION_DATE, strategy='mask', target_frequency='Q', delay_unit='D',
        reference_point='start', handle_missing_delays='ignore',
    ))),
    'panelwise-scaler': Case(_panel, _panelwise_scaler),
    'panelwise-delays': Case(_panel, _panelwise_delays),
    'imputer-interpolate-xy': Case(_reference_ts, _imputer_passes('interpolate', with_y=True)),
    'imputer-model-xy': Case(_reference_ts, _imputer_passes('model', with_y=True)),
    'imputer-tolerate-nan-panel': Case(_reference_panel, _imputer_passes('tolerate_nan', with_y=False), slow=True),
    'imputer-realistic-xy': Case(_xy_series, _imputer_passes('interpolate', with_y=True), slow=True),
    'drop-nan-target': Case(_xy_series, _drop_nan_target),
    'xy-pipeline': Case(_xy_series, _pipeline, slow=True),
    'out-of-sample-split': Case(_months, _split(lambda: OutOfSampleSplit(n_splits=3, test_size=2, gap=1))),
    'in-sample-split': Case(_months, _split(lambda: InSampleSplit(n_splits=2, test_size=3))),
    'ts-out-of-sample-split': Case(_months, _split(lambda: TSOutOfSampleSplit(n_splits=3))),
    'ts-in-sample-split': Case(_months, _split(lambda: TSInSampleSplit(n_splits=2, test_size=2))),
    'panel-out-of-sample-split': Case(_staggered, _split(lambda: PanelOutOfSampleSplit(n_splits=2, test_size=1),
                                                         with_groups=True)),
    'panel-in-sample-split': Case(_staggered, _split(lambda: PanelInSampleSplit(test_size=2), with_groups=True)),
    'panel-out-of-sample-per-entity': Case(_staggered, _split(
        lambda: PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=1))),
    'panel-in-sample-per-entity': Case(_staggered, _split(lambda: PanelInSampleSplitPerEntity(test_size=1))),
}


# =============================================================================
# Instantanés, comparaison, entrées en lecture seule
# =============================================================================
def snapshot(obj: Any) -> Any:
    """Deep copy of a pandas or NumPy object, its index frequency included; other objects as is."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.copy(deep=True), getattr(obj.index, 'freq', None)
    if isinstance(obj, np.ndarray):
        return obj.copy(), None
    return obj, None


def assert_unchanged(obj: Any, frozen: Any, label: str) -> None:
    """Assert that an object handed to the package equals its snapshot: values, dtypes, index, frequency.

    Raises:
        AssertionError: Naming ``label`` when the object was modified.
    """
    copy, freq = frozen
    if isinstance(obj, pd.DataFrame):
        pd.testing.assert_frame_equal(obj, copy, check_freq=True, obj=label)
    elif isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, copy, check_freq=True, obj=label)
    elif isinstance(obj, np.ndarray):
        np.testing.assert_array_equal(obj, copy, err_msg=label)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        assert getattr(obj.index, 'freq', None) == freq, f'{label}: index frequency changed'


def as_read_only(obj: Any) -> Any:
    """Rebuild a float ``DataFrame`` / ``Series`` on a read-only NumPy buffer; other objects unchanged.

    The returned object shares its values with a ``writeable=False`` array, so
    that any in-place write of the package into the values it receives raises
    ``ValueError: assignment destination is read-only``.
    """
    if isinstance(obj, pd.DataFrame) and len(obj.columns) and all(
        pd.api.types.is_float_dtype(dtype) for dtype in obj.dtypes
    ):
        values = obj.to_numpy(dtype=float, copy=True)
        values.flags.writeable = False
        frame = pd.DataFrame(values, index=obj.index, columns=obj.columns, copy=False)
        assert np.shares_memory(frame.to_numpy(), values)
        return frame
    if isinstance(obj, pd.Series) and pd.api.types.is_float_dtype(obj.dtype):
        values = obj.to_numpy(dtype=float, copy=True)
        values.flags.writeable = False
        series = pd.Series(values, index=obj.index, name=obj.name, copy=False)
        assert np.shares_memory(series.to_numpy(), values)
        return series
    return obj


def assert_same_outputs(actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Assert that two runs of a case gave the same outputs, key by key."""
    assert actual.keys() == expected.keys()
    for key in expected:
        a, e = actual[key], expected[key]
        if isinstance(e, pd.DataFrame):
            pd.testing.assert_frame_equal(a, e, obj=key)
        elif isinstance(e, pd.Series):
            pd.testing.assert_series_equal(a, e, obj=key)
        elif isinstance(e, np.ndarray):
            np.testing.assert_array_equal(a, e, err_msg=key)
        else:
            assert a == e, key


def guarded_run(case: Case, inputs: Dict[str, Any]) -> tuple:
    """Run a case, registering every object handed to the package.

    Returns:
        ``(outputs, registered)`` where ``registered`` lists ``(label, object, snapshot)``.
    """
    registered: List[tuple] = []

    def guard(obj):
        registered.append((f'object #{len(registered)} ({type(obj).__name__})', obj, snapshot(obj)))
        return obj

    outputs = case.run(inputs, guard)
    return outputs, registered
