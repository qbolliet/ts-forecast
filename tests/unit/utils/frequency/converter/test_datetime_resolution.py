"""Output resolution of ``FrequencyConverter``: the dates come back at the unit of the input.

Downsampling (``aggregate_to_lower_frequency``), upsampling
(``interpolate_to_higher_frequency``: extended grid, ``anchor_fraction``, sub-daily and
semi-monthly targets, ``asfreq`` fallback) and ``convert_frequency`` on a series, a frame
and a panel, with dates stored in seconds, milliseconds, microseconds or nanoseconds
(``with_datetime_resolution``): converted values and dates equal those of the nanosecond
input, and the dates come back at the input unit, one explicit line per path. A sub-second
grid of dates in seconds moves to milliseconds, never rounded; an upsampling followed by
the matching downsampling gives back the input index, unit included.
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


def _assert_same_conversion(result, reference) -> None:
    """Same values and dates (instants) as the reference, whatever the index unit."""
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, reference, check_index_type=False)
    else:
        pd.testing.assert_frame_equal(result, reference, check_index_type=False)


def _timeseries() -> pd.DataFrame:
    """Mixed-frequency time series of notebook 2 (monthly, quarterly and annual columns)."""
    return build_mixed_frequency_timeseries()


def _quarterly() -> pd.Series:
    """Quarterly GDP, observed rows only."""
    return _timeseries()['pib_trimestriel'].dropna()


def _panel() -> pd.DataFrame:
    """Three-country panel, monthly columns only."""
    return build_mixed_frequency_panel()[['inflation_ipc', 'taux_chomage']]


def _irregular() -> pd.Series:
    """Three irregular dates: no detectable source frequency (``asfreq`` fallback)."""
    return pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(['2024-01-01', '2024-02-15', '2024-05-01']))


def _daily() -> pd.Series:
    """Ten consecutive days."""
    return pd.Series(np.arange(10.0), index=pd.date_range('2024-01-01', periods=10, freq='D'), name='d')


# Chemins de conversion, une ligne par chemin : (jeu d'entrée, conversion). Chacun rend
# ses dates dans l'unité de l'entrée
PATHS = {
    'aggregate-series': (lambda: _timeseries()['inflation_ipc'],
                         lambda c, d: c.aggregate_to_lower_frequency(d, 'QS', method='mean')),
    'aggregate-frame-full-periods': (lambda: _timeseries()[['inflation_ipc', 'taux_chomage']],
                                     lambda c, d: c.aggregate_to_lower_frequency(d, 'QE', 'sum', full_periods_only=True)),
    'upsample-extended-grid': (_quarterly, lambda c, d: c.interpolate_to_higher_frequency(d, 'MS')),
    'upsample-anchor-fraction': (_quarterly,
                                 lambda c, d: c.interpolate_to_higher_frequency(d, 'MS', anchor_fraction=0.5)),
    'upsample-sub-daily-target': (_daily, lambda c, d: c.interpolate_to_higher_frequency(d, 'h')),
    'upsample-semi-monthly-target': (lambda: _timeseries()['inflation_ipc'].dropna(),
                                     lambda c, d: c.interpolate_to_higher_frequency(d, 'SMS')),
    'upsample-frame': (lambda: _timeseries()[['pib_trimestriel']].dropna(),
                       lambda c, d: c.interpolate_to_higher_frequency(d, 'MS')),
    'upsample-asfreq-fallback': (_irregular, lambda c, d: c.interpolate_to_higher_frequency(d, 'D')),
    'convert-series-up': (_quarterly, lambda c, d: c.convert_frequency(d, 'MS', method='linear')),
    'convert-series-down': (lambda: _timeseries()['inflation_ipc'], lambda c, d: c.convert_frequency(d, 'QS')),
    'convert-frame-up': (lambda: _timeseries()[['pib_trimestriel', 'balance_commerciale_annuelle']],
                         lambda c, d: c.convert_frequency(d, 'MS', method='linear')),
    'convert-frame-mixed-targets': (
        lambda: _timeseries()[['inflation_ipc', 'pib_trimestriel', 'balance_commerciale_annuelle']],
        lambda c, d: c.convert_frequency(d, {'pib_trimestriel': 'MS', 'balance_commerciale_annuelle': 'QS'},
                                         method='linear', alignment_method='nearest'),
    ),
    'convert-time-column': (lambda: _timeseries()[['inflation_ipc']].reset_index(),
                            lambda c, d: c.convert_frequency(d, 'QS', time_col='date')),
    'convert-panel-up': (_panel, lambda c, d: c.convert_frequency(d, 'W', method='linear')),
    'convert-panel-down': (_panel, lambda c, d: c.convert_frequency(d, 'QS')),
    'convert-panel-entity-target': (_panel,
                                    lambda c, d: c.convert_frequency(d, {('France',): 'W'}, method='linear')),
}


class TestDatetimeResolution:
    """Converted values and dates do not depend on how the input dates are stored."""

    # Le chemin « colonne de dates » avertit du remplacement de l'index par la validation
    @pytest.mark.filterwarnings('ignore:Index replaced with')
    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('path', list(PATHS), ids=list(PATHS))
    def test_dates_come_back_at_the_input_unit(self, path: str, unit: str) -> None:
        """Same conversion as in nanoseconds; every date of the output at the input unit."""
        build, convert = PATHS[path]
        reference = convert(FrequencyConverter(), with_datetime_resolution(build(), 'ns'))

        result = convert(FrequencyConverter(), with_datetime_resolution(build(), unit))

        _assert_same_conversion(result, reference)
        assert _units(result) == {unit}

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('method', ['mean', 'sum', 'last'])
    def test_aggregation_methods_keep_input_resolution(self, method: str, unit: str) -> None:
        """Monthly to quarterly: same totals as in nanoseconds, index at the input unit."""
        monthly = _timeseries()['inflation_ipc']
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
    def test_interpolation_methods_keep_input_resolution(self, method: str, unit: str) -> None:
        """Quarterly to monthly: same values and dates as in nanoseconds, grid at the input unit."""
        reference = FrequencyConverter().interpolate_to_higher_frequency(
            with_datetime_resolution(_quarterly(), 'ns'), 'MS', method=method
        )

        result = FrequencyConverter().interpolate_to_higher_frequency(
            with_datetime_resolution(_quarterly(), unit), 'MS', method=method
        )

        pd.testing.assert_series_equal(result, reference, check_index_type=False)
        assert _units(result) == {unit}

    @pytest.mark.parametrize('unit, expected', [('s', 'ms'), ('ms', 'ms'), ('us', 'us'), ('ns', 'ns')])
    def test_sub_second_grid_moves_to_the_finest_needed_unit(self, unit: str, expected: str) -> None:
        """Seconds interpolated every 100 ms: the grid needs milliseconds, never a rounding."""
        seconds = pd.Series([0.0, 1.0, 3.0], index=pd.date_range('2024-01-01', periods=3, freq='s'))

        result = FrequencyConverter().interpolate_to_higher_frequency(
            with_datetime_resolution(seconds, unit), '100ms'
        )

        assert result.index.unit == expected
        # Valeur d'or : pas de 100 ms, de 0 (00:00:00) à 1 (00:00:01) par dixièmes
        assert result.index[1] == pd.Timestamp('2024-01-01 00:00:00.1')
        assert result.iloc[:11].round(12).tolist() == [step / 10 for step in range(11)]

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_upsampling_then_downsampling_gives_back_the_input_index(self, unit: str) -> None:
        """Quarter starts to month starts and back: input index and values, unit included."""
        quarterly = with_datetime_resolution(
            pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range('2023-01-01', periods=4, freq='QS')), unit
        )

        monthly = FrequencyConverter().interpolate_to_higher_frequency(quarterly, 'MS')
        back = FrequencyConverter().aggregate_to_lower_frequency(monthly, 'QS', method='first')

        pd.testing.assert_index_equal(back.index, quarterly.index)
        assert back.index.unit == unit
        assert back.tolist() == quarterly.tolist()

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
