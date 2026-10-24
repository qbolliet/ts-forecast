"""Unit of the dates returned by ``HighFrequencyImputer``, whatever the resolution of the input.

The ``TS`` and ``PANEL-F`` reference datasets with dates stored in seconds,
milliseconds, microseconds or nanoseconds (``with_datetime_resolution``), under
the ``interpolate`` and ``model`` covariate strategies, without and with ``y``:
the stacked output, the provenance matrix, the replay of ``transform`` and the
inversion equal those of the nanosecond input, and every date they hold is at
the input unit - output and provenance at the same one, the inversion giving
back the input index unit included. The training pass ``fit_transform(X, y)``
and the prediction pass return ``X`` and ``y`` at the unit of ``X``, even for a
``y`` stored at another unit; the fitted state holds the unit of the fit input,
untouched by a ``transform`` on data stored at another unit, which returns the
unit of those data. Under ``XYPipeline(memory=...)``, two units of the same
instants are two cache entries, each served at its own unit.
"""

# Modules de base
import warnings
from collections.abc import Mapping

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Perturbations et estimateurs partagés
from tests.support.estimators import SpyEstimator
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.xy import DropNaNTarget, XYPipeline

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import make_imputer, public_fitted_state

# Sorties d'une passe complète, dans l'ordre où elles sont produites
PASS_OUTPUTS = ('output', 'provenance', 'replay', 'inverse')


def _passes(data: pd.DataFrame, strategy: str) -> dict:
    """Fit an imputer on ``data`` and return its output, provenance, replay and inversion, and the imputer."""
    imputer = make_imputer(covariate_strategy=strategy, restore_original_values=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        output = imputer.fit_transform(data)
        provenance = imputer.imputation_provenance_.copy()
        replay = imputer.transform(data)
        inverse = imputer.inverse_transform(output)
    return {'output': output, 'provenance': provenance, 'replay': replay, 'inverse': inverse, 'imputer': imputer}


def _cached_passes(cache: dict, dataset: tuple, strategy: str, unit: str) -> dict:
    """Passes of :func:`_passes` on ``dataset`` stored at ``unit``, computed once per module."""
    name, data = dataset
    key = (name, strategy, unit)
    if key not in cache:
        cache[key] = _passes(with_datetime_resolution(data, unit), strategy)
    return cache[key]


def _all_at(obj, unit: str) -> dict:
    """``datetime_units`` of ``obj`` as the policy expects it: every date location at ``unit``."""
    return {location: unit for location in datetime_units(obj)}


def _state_units(value, path: str = 'state') -> dict:
    """Map every date held by a fitted state (frames, indexes, timestamps, nested containers) to its unit."""
    units = {}
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        units.update({f'{path}{location}': unit for location, unit in datetime_units(value).items()})
    elif isinstance(value, pd.Timestamp):
        units[path] = value.unit
    elif isinstance(value, Mapping):
        for key, item in value.items():
            units.update(_state_units(item, f'{path}[{key!r}]'))
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            units.update(_state_units(item, f'{path}[{position}]'))
    return units


def _xy(data: pd.DataFrame, x_unit: str, y_unit: str) -> tuple:
    """``(X, y)`` of the ``TS`` dataset, ``y = a2``, with dates stored at two units."""
    return (
        with_datetime_resolution(data.drop(columns='a2'), x_unit),
        with_datetime_resolution(data['a2'], y_unit),
    )


@pytest.fixture(params=['TS', pytest.param('PANEL-F', marks=pytest.mark.slow)])
def reference_dataset(request, reference_timeseries, mixed_freq_panel_multifrequency) -> tuple:
    """``(name, data)``: the ``TS`` series and the ``PANEL-F`` panel (heterogeneous frequency of ``v``)."""
    data = reference_timeseries if request.param == 'TS' else mixed_freq_panel_multifrequency
    return request.param, data


@pytest.fixture(scope='module')
def passes_cache() -> dict:
    """Cache of the passes, computed once per (dataset, strategy, unit) and shared by the tests."""
    return {}


class TestDatetimeResolution:
    """Imputed values, provenance and grid dates do not depend on how the dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_passes_do_not_depend_on_resolution(self, reference_dataset, passes_cache, strategy, unit):
        """Output, provenance, replay and inversion equal the nanosecond ones, at the input unit."""
        expected = _cached_passes(passes_cache, reference_dataset, strategy, 'ns')

        result = _cached_passes(passes_cache, reference_dataset, strategy, unit)

        for output in PASS_OUTPUTS:
            pd.testing.assert_frame_equal(result[output], expected[output], check_index_type=False, obj=output)
        # Toute date rendue (index, niveaux temporels) à l'unité de l'entrée
        for output in PASS_OUTPUTS:
            units = datetime_units(result[output])
            assert units and units == _all_at(result[output], unit), output

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_output_and_provenance_share_their_index(self, reference_dataset, passes_cache, unit):
        """The provenance matrix is indexed exactly like the output, date unit included."""
        result = _cached_passes(passes_cache, reference_dataset, 'model', unit)

        pd.testing.assert_index_equal(result['provenance'].index, result['output'].index, exact=True)
        assert datetime_units(result['provenance']) == datetime_units(result['output'])

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_inverse_gives_back_the_input_index(self, reference_dataset, passes_cache, unit):
        """``inverse_transform(fit_transform(X))`` is indexed like ``X``, unit included."""
        stored = with_datetime_resolution(reference_dataset[1], unit)

        result = _cached_passes(passes_cache, reference_dataset, 'interpolate', unit)

        pd.testing.assert_index_equal(result['inverse'].index, stored.index, exact=True)
        pd.testing.assert_frame_equal(result['inverse'], stored)


class TestTargetPasses:
    """``X`` and ``y`` come out at the unit of ``X``, in the training and in the prediction pass."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_training_pass_with_target(self, reference_timeseries, unit):
        """``fit_transform(X, y)`` with ``y = a2`` gives the same augmented ``X`` and ``y``, at the input unit."""
        def run(data_unit):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return make_imputer().fit_transform(*_xy(reference_timeseries, data_unit, data_unit))

        expected_X, expected_y = run('ns')

        X_out, y_out = run(unit)

        pd.testing.assert_frame_equal(X_out, expected_X, check_index_type=False)
        pd.testing.assert_series_equal(y_out, expected_y, check_index_type=False)
        assert datetime_units(X_out) == _all_at(X_out, unit)
        assert datetime_units(y_out.index) == datetime_units(X_out.index)

    @pytest.mark.parametrize('x_unit, y_unit', [('s', 'ns'), ('ns', 's'), ('ms', 'us')])
    def test_target_at_another_unit_follows_x(self, reference_timeseries, x_unit, y_unit):
        """A ``y`` stored at another unit: same values as at ``X``'s unit, every pass at ``X``'s unit."""
        imputer = make_imputer(restore_original_values=True)
        X, y = _xy(reference_timeseries, x_unit, y_unit)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected_X, expected_y = make_imputer().fit_transform(*_xy(reference_timeseries, x_unit, x_unit))

            X_train, y_train = imputer.fit_transform(X, y)
            provenance = imputer.imputation_provenance_.copy()
            X_back, y_back = imputer.inverse_transform(X_train, y_train)
            X_pred, y_pred = imputer.transform(X, y)

        # Passe d'entraînement : celle d'un y stocké à l'unité de X, index compris
        pd.testing.assert_frame_equal(X_train, expected_X)
        pd.testing.assert_series_equal(y_train, expected_y)
        assert datetime_units(provenance) == datetime_units(X_train)
        # Inversion et passe de prédiction : lignes et unité de X, valeurs de y inchangées
        pd.testing.assert_index_equal(y_back.index, X.index, exact=True)
        pd.testing.assert_index_equal(X_pred.index, X.index, exact=True)
        pd.testing.assert_series_equal(y_pred, y.set_axis(X.index))


class TestFittedState:
    """The fitted state holds one unit, the one of the fit input, from ``fit`` to ``transform``."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_fitted_state_is_at_the_fit_unit(self, reference_dataset, passes_cache, unit):
        """Window masks and bounds, and every other date of the public fitted state, at the fit unit.

        The imputer has run its transforms and inversion since its fit, which
        never rewrite the fitted state.
        """
        imputer = _cached_passes(passes_cache, reference_dataset, 'model', unit)['imputer']

        units = _state_units(public_fitted_state(imputer))

        assert units and set(units.values()) == {unit}

    @pytest.mark.parametrize('fit_unit, transform_unit', [('s', 'ns'), ('ns', 's')])
    def test_transform_at_another_unit(self, reference_dataset, passes_cache, fit_unit, transform_unit):
        """Fitted at one unit, transformed at another: values of a refit, dates of the transformed data.

        The fitted state keeps the unit of the fit: the transform replays the
        plan on the dates of its own input and leaves the fit untouched.
        """
        _, data = reference_dataset
        imputer = make_imputer(covariate_strategy='model', restore_original_values=True)
        transformed_data = with_datetime_resolution(data, transform_unit)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(with_datetime_resolution(data, fit_unit))
            output = imputer.transform(transformed_data)
            provenance = imputer.imputation_provenance_.copy()
            inverse = imputer.inverse_transform(output)
        expected = _cached_passes(passes_cache, reference_dataset, 'model', transform_unit)['output']

        pd.testing.assert_frame_equal(output, expected)
        assert datetime_units(provenance) == datetime_units(output) == _all_at(output, transform_unit)
        pd.testing.assert_index_equal(inverse.index, transformed_data.index, exact=True)
        assert set(_state_units(public_fitted_state(imputer)).values()) == {fit_unit}


class CountingImputer(HighFrequencyImputer):
    """``HighFrequencyImputer`` counting, in a class counter, the fits really computed.

    A step loaded from the cache of ``XYPipeline(memory=...)`` is not fitted,
    so it leaves the counter unchanged.
    """

    n_fits = 0

    def fit(self, X, y=None):
        """Count the fit, then fit the imputer."""
        CountingImputer.n_fits += 1
        return super().fit(X, y)


class TestCachedPipeline:
    """Under ``XYPipeline(memory=...)``, each unit of the input dates is its own cache entry."""

    def test_each_unit_is_its_own_cache_entry(self, tmp_path, reference_timeseries):
        """Fits at ``'s'``, ``'ns'``, then ``'s'`` again: two imputer fits, each served at its unit.

        The cache key reads the dtypes of the data, date unit included: the
        same instants at two units are two keys. Outputs stored at the unit of
        the input differ by their dtypes too, so a key shared by the two units
        would serve one of them at the other's unit.
        """
        CountingImputer.n_fits = 0
        received = {}
        for unit in ('s', 'ns', 's'):
            pipeline = XYPipeline([
                ('impute', CountingImputer(target_frequency='M', estimator=LinearRegression())),
                ('drop', DropNaNTarget()),
                ('model', SpyEstimator()),
            ], memory=str(tmp_path))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipeline.fit(*_xy(reference_timeseries, unit, unit))
            received.setdefault(unit, []).append(pipeline['model'].fit_X_)

        assert CountingImputer.n_fits == 2
        # Chaque unité reçoit sa propre sortie : mêmes valeurs, dates à son unité
        assert datetime_units(received['s'][0]) == _all_at(received['s'][0], 's')
        assert datetime_units(received['ns'][0]) == _all_at(received['ns'][0], 'ns')
        pd.testing.assert_frame_equal(received['s'][1], received['s'][0])
        pd.testing.assert_frame_equal(received['ns'][0], received['s'][0], check_index_type=False)
