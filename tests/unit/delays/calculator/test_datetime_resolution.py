"""Independence of ``calculate_applicable_delay`` from the resolution of the delay table dates.

The detected delays of a series and of a two-country panel, whose date columns
(``observation_date``, ``download_date``, ``period_start``, ``period_end``) are
stored in seconds, milliseconds, microseconds or nanoseconds
(``with_datetime_resolution``): the applicable delays, for every reference point,
target frequency and target unit, equal those of the nanosecond table.
"""

import pandas as pd
import pytest

# Perturbations partagées
from tests.support.perturbations import DATETIME_UNITS, with_datetime_resolution

# Fonction à tester
from tsforecast.delays.calculator import calculate_applicable_delay


class TestDatetimeResolution:
    """Applicable delays do not depend on how the dates of the delay table are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('target_unit', [None, 'D', 's', 'us'])
    @pytest.mark.parametrize('frequency', ['M', 'Q', 'Y'])
    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_panel_delays(self, monthly_publication_delays, reference_point, frequency, target_unit, unit):
        """Panel delays converted to another reference point, frequency and unit: same values."""
        reference = calculate_applicable_delay(
            with_datetime_resolution(monthly_publication_delays, 'ns'), reference_point, frequency,
            unit=target_unit, aggregate_by_panel=True,
        )

        result = calculate_applicable_delay(
            with_datetime_resolution(monthly_publication_delays, unit), reference_point, frequency,
            unit=target_unit, aggregate_by_panel=True,
        )

        pd.testing.assert_frame_equal(result, reference)

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_series_delay_golden_value(self, simple_time_series_delays, unit):
        """December 2023 published on 2024-01-15: 14 days after the end of the month, at every resolution."""
        # Valeur d'or : fin de décembre = 1er janvier 2024 (borne exclusive) ; 15 - 1 = 14 jours
        delays = with_datetime_resolution(simple_time_series_delays, unit)

        result = calculate_applicable_delay(delays, 'end', 'M', unit='D', indicators=['PIB'])

        assert result.loc['PIB', 'delay'] == 14
