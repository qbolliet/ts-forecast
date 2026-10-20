"""Independence of ``compare_and_detect_delays`` from the resolution of the input dates.

Series and panels whose dates are stored in seconds, milliseconds, microseconds
or nanoseconds (``with_datetime_resolution``), the two downloads possibly at
different resolutions: the detected observations and their delays (in days,
seconds and microseconds, from the start or the end of the period) equal those
of the nanosecond input; ``observation_date`` keeps the input resolution, the
period bounds and the download date have one resolution whatever the input.
"""

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support.datasets import HETEROGENEOUS_PANEL_COUNTRIES, build_mixed_frequency_panel
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import _series


def _older(data: pd.DataFrame) -> tuple:
    """Copy of a panel where the last observation of every observed (entity, column) is missing.

    Returns:
        ``(older, n_blanked)``: the earlier snapshot and the number of observations removed.
    """
    older = data.copy()
    n_blanked = 0
    for entity in data.index.get_level_values(0).unique():
        block = data.xs(entity, level=0)
        for column in data.columns:
            observed = block[column].dropna()
            # Couple jamais observé (colonne structurellement absente) : rien à retirer
            if observed.empty:
                continue
            older.loc[(entity, observed.index[-1]), column] = np.nan
            n_blanked += 1
    return older, n_blanked


@pytest.fixture(scope='module')
def realistic_panel() -> pd.DataFrame:
    """Heterogeneous panel of notebook 3 (monthly, quarterly and annual columns)."""
    return build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)


class TestDatetimeResolution:
    """Detected delays do not depend on how the dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('delay_unit', ['D', 's', 'us'])
    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_single_download(self, realistic_panel, reference_point, delay_unit, unit):
        """First download of the realistic panel: same rows and delays as in nanoseconds."""
        reference = compare_and_detect_delays(
            with_datetime_resolution(realistic_panel, 'ns'), None, '2024-10-15 13:45:10.123456',
            reference_point=reference_point, delay_unit=delay_unit,
        )

        result = compare_and_detect_delays(
            with_datetime_resolution(realistic_panel, unit), None, '2024-10-15 13:45:10.123456',
            reference_point=reference_point, delay_unit=delay_unit,
        )

        pd.testing.assert_frame_equal(result, reference, check_dtype=False)
        # Date d'observation : résolution de l'entrée ; bornes de période et téléchargement :
        # une seule résolution, quelle que soit l'entrée
        units = datetime_units(result)
        assert units.pop(('column', 'observation_date')) == unit
        assert units == {key: value for key, value in datetime_units(reference).items()
                         if key != ('column', 'observation_date')}

    @pytest.mark.parametrize('new_unit', DATETIME_UNITS)
    @pytest.mark.parametrize('existing_unit', ['s', 'ns'])
    def test_two_downloads_at_different_resolutions(self, realistic_panel, existing_unit, new_unit):
        """The two snapshots are compared by instant, not by stored integer."""
        older, n_blanked = _older(realistic_panel)
        reference = compare_and_detect_delays(
            with_datetime_resolution(realistic_panel, 'ns'), with_datetime_resolution(older, 'ns'), '2024-10-15'
        )

        result = compare_and_detect_delays(
            with_datetime_resolution(realistic_panel, new_unit), with_datetime_resolution(older, existing_unit),
            '2024-10-15',
        )

        pd.testing.assert_frame_equal(result, reference, check_dtype=False)
        # Une observation détectée par observation retirée de l'instantané antérieur
        assert len(result) == n_blanked

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_microsecond_delay_golden_value(self, unit):
        """April observation downloaded on 2023-05-10 06:00: 39 days and 6 hours after April 1st."""
        # Valeur d'or : (39 x 86 400 + 6 x 3 600) s = 3 391 200 s = 3 391 200 000 000 µs
        series = with_datetime_resolution(_series([1.0, 2.0, 3.0, 4.0]), unit)

        result = compare_and_detect_delays(series, None, '2023-05-10 06:00', delay_unit='us')

        assert result['delay'].tolist() == [3_391_200_000_000.0]
        assert result['observation_date'].tolist() == [pd.Timestamp('2023-04-01')]
