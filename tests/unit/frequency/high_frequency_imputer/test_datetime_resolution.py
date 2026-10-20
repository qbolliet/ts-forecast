"""Independence of ``HighFrequencyImputer`` from the resolution of the input dates.

The ``TS`` and ``PANEL-F`` reference datasets with dates stored in seconds,
milliseconds, microseconds or nanoseconds (``with_datetime_resolution``), under
the ``interpolate`` and ``model`` covariate strategies, without and with ``y``:
the stacked output, the provenance matrix, the replay of ``transform`` and the
inversion equal those of the nanosecond input. The output grid has one
resolution whatever the input (the grid is built by the imputer); the
provenance matrix of a panel keeps the input resolution.
"""

# Modules de base
import warnings

import pandas as pd
import pytest

# Perturbations partagées
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import make_imputer


def _passes(data: pd.DataFrame, strategy: str) -> dict:
    """Fit an imputer on ``data`` and return its output, provenance, replay and inversion."""
    imputer = make_imputer(covariate_strategy=strategy, restore_original_values=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        output = imputer.fit_transform(data)
        provenance = imputer.imputation_provenance_.copy()
        replay = imputer.transform(data)
        inverse = imputer.inverse_transform(output)
    return {'output': output, 'provenance': provenance, 'replay': replay, 'inverse': inverse}


@pytest.fixture(params=['TS', pytest.param('PANEL-F', marks=pytest.mark.slow)])
def reference_dataset(request, reference_timeseries, mixed_freq_panel_multifrequency) -> tuple:
    """``(name, data)``: the ``TS`` series and the ``PANEL-F`` panel (heterogeneous frequency of ``v``)."""
    data = reference_timeseries if request.param == 'TS' else mixed_freq_panel_multifrequency
    return request.param, data


@pytest.fixture(scope='module')
def nanosecond_passes() -> dict:
    """Cache of the nanosecond passes, computed once per (dataset, strategy)."""
    return {}


class TestDatetimeResolution:
    """Imputed values, provenance and grid dates do not depend on how the dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_passes_do_not_depend_on_resolution(self, reference_dataset, nanosecond_passes, strategy, unit):
        """Output, provenance, replay and inversion equal the nanosecond ones, one grid unit."""
        name, data = reference_dataset
        key = (name, strategy)
        if key not in nanosecond_passes:
            nanosecond_passes[key] = _passes(with_datetime_resolution(data, 'ns'), strategy)
        expected = nanosecond_passes[key]

        result = _passes(with_datetime_resolution(data, unit), strategy)

        for output in ('output', 'provenance', 'replay'):
            pd.testing.assert_frame_equal(result[output], expected[output], check_index_type=False, obj=output)
        pd.testing.assert_frame_equal(result['inverse'], expected['inverse'], check_index_type=False)
        # Sortie empilée et rejeu sur la grille construite par l'imputeur : une unité fixe,
        # celle obtenue depuis des nanosecondes ; matrice de provenance d'un panel, indexée
        # sur les dates de l'entrée : résolution de l'entrée
        assert datetime_units(result['output']) == datetime_units(expected['output'])
        assert datetime_units(result['replay']) == datetime_units(expected['replay'])
        provenance_units = set(datetime_units(result['provenance']).values())
        assert provenance_units == (
            set(datetime_units(expected['provenance']).values()) if name == 'TS' else {unit}
        )

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_training_pass_with_target(self, reference_timeseries, unit):
        """``fit_transform(X, y)`` with ``y = a2`` gives the same augmented ``X`` and ``y``."""
        def run(data):
            X, y = data.drop(columns='a2'), data['a2']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return make_imputer().fit_transform(X, y)

        expected_X, expected_y = run(with_datetime_resolution(reference_timeseries, 'ns'))

        X_out, y_out = run(with_datetime_resolution(reference_timeseries, unit))

        pd.testing.assert_frame_equal(X_out, expected_X, check_index_type=False)
        pd.testing.assert_series_equal(y_out, expected_y, check_index_type=False)
