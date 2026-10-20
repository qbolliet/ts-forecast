"""Independence of ``FrequencyConverter`` from the resolution of the input dates.

Downsampling (``aggregate_to_lower_frequency``), upsampling
(``interpolate_to_higher_frequency``) and ``convert_frequency`` on a series and
on a panel, with dates stored in seconds, milliseconds, microseconds or
nanoseconds (``with_datetime_resolution``): converted values and dates equal
those of the nanosecond input. An aggregated series or panel keeps the input
resolution; the upsampled grid of a series has one resolution whatever the input,
that of a panel follows the input from the microsecond on.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.datasets import build_mixed_frequency_panel, build_mixed_frequency_timeseries
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution
from tsforecast.utils.frequency.converter import FrequencyConverter


def _units(result) -> set:
    """Resolutions of the dates of a converted object."""
    return set(datetime_units(result).values())


class TestDatetimeResolution:
    """Converted values and dates do not depend on how the input dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('method', ['mean', 'sum', 'last'])
    def test_aggregation_keeps_input_resolution(self, method: str, unit: str) -> None:
        """Monthly to quarterly: same totals as in nanoseconds, index at the input resolution."""
        monthly = build_mixed_frequency_timeseries()['inflation_ipc']
        reference = FrequencyConverter().aggregate_to_lower_frequency(
            with_datetime_resolution(monthly, 'ns'), 'QS', method=method
        )

        result = FrequencyConverter().aggregate_to_lower_frequency(
            with_datetime_resolution(monthly, unit), 'QS', method=method
        )

        pd.testing.assert_series_equal(result, reference, check_index_type=False)
        assert _units(result) == {unit}

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('method', ['linear', 'time', 'nearest'])
    def test_upsampling_grid_does_not_follow_input_resolution(self, method: str, unit: str) -> None:
        """Quarterly to monthly: same values and dates as in nanoseconds; one grid unit for every input."""
        quarterly = build_mixed_frequency_timeseries()['pib_trimestriel'].dropna()
        reference = FrequencyConverter().interpolate_to_higher_frequency(
            with_datetime_resolution(quarterly, 'ns'), 'MS', method=method
        )

        result = FrequencyConverter().interpolate_to_higher_frequency(
            with_datetime_resolution(quarterly, unit), 'MS', method=method
        )

        pd.testing.assert_series_equal(result, reference, check_index_type=False)
        # Grille sur-échantillonnée construite par le convertisseur : une seule unité,
        # celle obtenue depuis une entrée en nanosecondes
        assert _units(result) == _units(reference)

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize(
        'target, method', [('QS', 'mean'), ('W', 'linear')], ids=['downsampling', 'upsampling']
    )
    def test_panel_conversion(self, target: str, method: str, unit: str) -> None:
        """Every entity of the panel is converted as in nanoseconds, monthly columns only."""
        panel = build_mixed_frequency_panel()[['inflation_ipc', 'taux_chomage']]
        reference = FrequencyConverter().convert_frequency(
            with_datetime_resolution(panel, 'ns'), target, method=method
        )

        result = FrequencyConverter().convert_frequency(
            with_datetime_resolution(panel, unit), target, method=method
        )

        pd.testing.assert_frame_equal(result, reference, check_index_type=False)
        # Panel : à la descente, périodes datées sur les dates de l'entrée (résolution de
        # l'entrée) ; à la montée, grille tirée par pandas des dates de l'entrée, au moins
        # à la microseconde (résolution par défaut de pandas)
        if target == 'QS':
            assert _units(result) == {unit}
        else:
            assert _units(result) == {'ns' if unit == 'ns' else 'us'}

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_values_of_an_upsampling_are_exact(self, unit: str) -> None:
        """Linear interpolation between two quarter starts: golden values at every resolution."""
        # Valeur d'or : 0 en janvier, 3 en avril, interpolation linéaire au pas mensuel
        # (méthode 'linear' : en positions, pas en jours) -> 0, 1, 2, 3
        quarterly = pd.Series([0.0, 3.0], index=pd.DatetimeIndex(['2024-01-01', '2024-04-01']).as_unit(unit))

        result = FrequencyConverter().interpolate_to_higher_frequency(quarterly, 'MS', method='linear')

        assert result.loc['2024-01-01':'2024-04-01'].tolist() == [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_array_equal(
            result.index[:4].as_unit('ns').asi8,
            pd.date_range('2024-01-01', periods=4, freq='MS').as_unit('ns').asi8,
        )
