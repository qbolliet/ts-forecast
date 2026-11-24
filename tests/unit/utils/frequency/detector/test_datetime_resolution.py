"""Independence of frequency detection from the resolution of the input dates.

Covers ``detect_frequency``, ``detect_dataset_frequency`` and
``detect_index_frequency`` on a regular series, the genuinely irregular series
and the heterogeneous panel of notebook 3, with dates stored in seconds,
milliseconds, microseconds or nanoseconds (``with_datetime_resolution``), and on
sub-daily grids whose spacing is a whole number of the coarsest resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_mixed_frequency_panel,
    build_mixed_frequency_timeseries,
)
from tests.support.perturbations import DATETIME_UNITS, with_datetime_resolution
from tsforecast.utils.frequency.utils import (
    detect_dataset_frequency,
    detect_frequency,
    detect_index_frequency,
)

from tests.unit.utils.frequency.detector.helpers import RETURN_FORMATS

# Jeux de détection : série régulière, série irrégulière, panel hétérogène
_DATASETS = {
    'regular_series': build_mixed_frequency_timeseries,
    'irregular_series': lambda: build_mixed_frequency_timeseries(annual_start_date='2015-01-01'),
    'heterogeneous_panel': lambda: build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES),
}


class TestDatetimeResolution:
    """Detected frequencies do not depend on how the dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('return_format', RETURN_FORMATS)
    @pytest.mark.parametrize('dataset', list(_DATASETS), ids=list(_DATASETS))
    def test_dataset_frequencies(self, dataset: str, return_format: str, unit: str) -> None:
        """Each column (or (entity, column) couple) gets the frequency detected in nanoseconds."""
        data = _DATASETS[dataset]()
        expected = detect_dataset_frequency(with_datetime_resolution(data, 'ns'), return_format=return_format)

        result = detect_dataset_frequency(with_datetime_resolution(data, unit), return_format=return_format)

        assert result == expected

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('dataset', list(_DATASETS), ids=list(_DATASETS))
    def test_index_and_global_frequency(self, dataset: str, unit: str) -> None:
        """The frequency of the index and the global frequency match the nanosecond detection."""
        data = _DATASETS[dataset]()
        reference = with_datetime_resolution(data, 'ns')
        converted = with_datetime_resolution(data, unit)

        assert detect_index_frequency(converted.index, return_format='full') == \
            detect_index_frequency(reference.index, return_format='full')
        assert detect_frequency(converted, return_format='full') == \
            detect_frequency(reference, return_format='full')

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize(
        'spacing, expected',
        [('1h', 'h'), ('15min', '15min'), ('30s', '30s'), ('36h', '36h')],
        ids=['hour', 'quarter-hour', 'thirty-seconds', 'thirty-six-hours'],
    )
    def test_sub_daily_spacing(self, spacing: str, expected: str, unit: str) -> None:
        """A sub-daily grid gives the same alias at every resolution that holds it exactly."""
        # Grille de 20 dates, écart multiple entier de la seconde : représentable en 's'
        dates = pd.date_range('2024-03-01', periods=20, freq=spacing).as_unit(unit)
        series = pd.Series(np.arange(20.0), index=dates)

        assert detect_frequency(series, return_format='full') == expected
