"""Unit of the dates returned by ``compare_and_detect_delays``, whatever the resolution of the input.

Series and panels whose dates are stored in seconds, milliseconds, microseconds
or nanoseconds (``with_datetime_resolution``), the two downloads possibly at
different resolutions: the detected observations and their delays (in days,
seconds and microseconds, from the start or the end of the period) equal those
of the nanosecond input, and every date column of the result (observation,
period bounds, download) is stored at the unit of ``new_data``, finer only for
a download date that needs it; the same holds for a download date given as a
``Timestamp``, for time-zone-aware data, for undetected frequencies, for the
detection report and for empty results, whose columns and index levels carry the dtypes of
a non-empty result. The applicable delays derived from the result do not depend
on the resolution either.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support.datasets import HETEROGENEOUS_PANEL_COUNTRIES, build_mixed_frequency_panel
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution

# Fonctions à tester
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import OUTPUT_COLUMNS, _panel, _series

# Colonnes de dates du résultat
DATE_COLUMNS = ('observation_date', 'download_date', 'period_start', 'period_end')

# Date de téléchargement à la microseconde, et unité attendue de sa colonne selon
# l'unité des données : celle des données, sauf si elle ne peut pas la représenter
MICROSECOND_DOWNLOAD = '2024-10-15 13:45:10.123456'
MICROSECOND_DOWNLOAD_UNIT = {'s': 'us', 'ms': 'us', 'us': 'us', 'ns': 'ns'}


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


def _date_units(unit: str, download_unit: str = None) -> dict:
    """Expected ``datetime_units`` of a result: every date column at ``unit``, the download at ``download_unit``."""
    units = {('column', column): unit for column in DATE_COLUMNS}
    units[('column', 'download_date')] = download_unit or unit
    return units


def _small(structure: str, unit: str, tz: str = None) -> pd.DataFrame:
    """Four monthly observations of ``PIB``, as a series or a two-entity panel, at ``unit``."""
    if structure == 'series':
        data = _series([1.0, 2.0, 3.0, 4.0])
    else:
        data = _panel({'FR': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'DE': {'PIB': [5.0, 6.0, 7.0, 8.0]}})
    if tz is not None:
        data = data.tz_localize(tz, level=-1) if structure == 'panel' else data.tz_localize(tz)
    return with_datetime_resolution(data, unit)


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
            with_datetime_resolution(realistic_panel, 'ns'), None, MICROSECOND_DOWNLOAD,
            reference_point=reference_point, delay_unit=delay_unit,
        )

        result = compare_and_detect_delays(
            with_datetime_resolution(realistic_panel, unit), None, MICROSECOND_DOWNLOAD,
            reference_point=reference_point, delay_unit=delay_unit,
        )

        pd.testing.assert_frame_equal(result, reference, check_dtype=False)
        # Dates à l'unité de l'entrée ; téléchargement à la microseconde au plus grossier
        assert datetime_units(result) == _date_units(unit, MICROSECOND_DOWNLOAD_UNIT[unit])

    @pytest.mark.parametrize('new_unit', DATETIME_UNITS)
    @pytest.mark.parametrize('existing_unit', ['s', 'ns'])
    def test_two_downloads_at_different_resolutions(self, realistic_panel, existing_unit, new_unit):
        """The two snapshots are compared by instant; the dates follow ``new_data`` alone."""
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
        # Unité de new_data, même face à un existing_data plus fin
        assert datetime_units(result) == _date_units(new_unit)

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_microsecond_delay_golden_value(self, unit):
        """April observation downloaded on 2023-05-10 06:00: 39 days and 6 hours after April 1st."""
        # Valeur d'or : (39 x 86 400 + 6 x 3 600) s = 3 391 200 s = 3 391 200 000 000 µs
        series = with_datetime_resolution(_series([1.0, 2.0, 3.0, 4.0]), unit)

        result = compare_and_detect_delays(series, None, '2023-05-10 06:00', delay_unit='us')

        assert result['delay'].tolist() == [3_391_200_000_000.0]
        assert result['observation_date'].tolist() == [pd.Timestamp('2023-04-01')]


class TestOutputUnit:
    """Every date column of the result is stored at the unit of ``new_data``."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('download_date', [
        pd.Timestamp('2023-06-15').as_unit('s'),
        pd.Timestamp('2023-06-15').as_unit('ns'),
        datetime(2023, 6, 15),
    ], ids=['timestamp_s', 'timestamp_ns', 'datetime'])
    def test_download_date_follows_the_data(self, download_date, unit):
        """Whatever its own unit, the download date is returned at the unit of the data."""
        result = compare_and_detect_delays(_small('series', unit), None, download_date)

        assert datetime_units(result) == _date_units(unit)
        assert result['download_date'].tolist() == [pd.Timestamp('2023-06-15')]

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
    def test_download_date_finer_than_the_data_is_not_rounded(self, unit):
        """A download date with nanoseconds moves its column to ``'ns'``; the other dates keep the data unit."""
        given = pd.Timestamp('2023-06-15 12:30:45.123456789')

        result = compare_and_detect_delays(_small('series', unit), None, given)

        assert datetime_units(result) == _date_units(unit, 'ns')
        assert result['download_date'].iloc[0].value == given.value

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_string_download_date_keeps_its_nanoseconds(self, unit):
        """A download date given as a string with nanoseconds is read like the equivalent ``Timestamp``."""
        text = '2023-05-10 00:00:00.000000001'
        data = _small('series', unit)

        from_string = compare_and_detect_delays(data, None, text, delay_unit='us')
        from_timestamp = compare_and_detect_delays(data, None, pd.Timestamp(text), delay_unit='us')

        pd.testing.assert_frame_equal(from_string, from_timestamp)
        # Valeur d'or : 1er avril -> 10 mai = 39 j = 3 369 600 000 000 µs, plus 1 ns arrondie par excès à 1 µs
        assert from_string['delay'].tolist() == [3_369_600_000_001.0]
        assert from_string['download_date'].iloc[0].value == pd.Timestamp(text).value

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_download_date_keeps_its_time_zone(self, unit):
        """An aware download date keeps its zone; naive data keep none."""
        given = pd.Timestamp('2023-06-15 08:00', tz='Asia/Tokyo')

        result = compare_and_detect_delays(_small('series', unit), None, given)

        assert str(result['download_date'].dtype) == f'datetime64[{unit}, Asia/Tokyo]'
        assert str(result['observation_date'].dtype) == f'datetime64[{unit}]'

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('structure', ['series', 'panel'])
    def test_time_zone_aware_data(self, structure, unit):
        """Observation and period bounds keep the zone and the unit of the data."""
        result = compare_and_detect_delays(_small(structure, unit, tz='Europe/Paris'), None, '2023-06-15')

        for column in ('observation_date', 'period_start', 'period_end'):
            assert str(result[column].dtype) == f'datetime64[{unit}, Europe/Paris]', column

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('tz', [None, 'Europe/Paris'], ids=['naive', 'aware'])
    def test_undetected_frequency_keeps_the_dtype_of_the_bounds(self, tz, unit):
        """A single observation has no frequency: its missing bounds are typed like detected ones."""
        data = _small('series', unit, tz=tz)
        once = data.assign(PIB=[np.nan, 2.0, np.nan, np.nan])
        detected = compare_and_detect_delays(data, None, '2023-06-15')

        with pytest.warns(UserWarning, match='could not be detected'):
            result = compare_and_detect_delays(once, None, '2023-06-15')

        assert result['period_start'].isna().all() and result['period_end'].isna().all()
        pd.testing.assert_series_equal(result.dtypes, detected.dtypes)

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_report_carries_the_returned_download_date(self, unit):
        """The report holds the download date of the result, at the same unit."""
        result, report = compare_and_detect_delays(
            _small('panel', unit), None, MICROSECOND_DOWNLOAD, return_report=True
        )

        assert report.download_date == result['download_date'].iloc[0]
        assert report.download_date.unit == MICROSECOND_DOWNLOAD_UNIT[unit]
        assert report.to_dict()['download_date'] == '2024-10-15T13:45:10.123456'

    @pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
    def test_infra_second_grid(self, unit):
        """A grid of 100 ms: observation and bounds at the data unit, which holds them."""
        data = pd.DataFrame({'x': np.arange(6.0)}, index=pd.date_range('2024-01-01', periods=6, freq='100ms'))

        result = compare_and_detect_delays(
            with_datetime_resolution(data, unit), None, '2024-01-01 00:00:01.25', delay_unit='us'
        )

        assert datetime_units(result) == _date_units(unit)
        # Valeur d'or : 1,25 s - 0,5 s (début de la dernière période) = 750 000 µs
        assert result['delay'].tolist() == [750_000.0]

    @pytest.mark.filterwarnings('ignore:Index replaced with')
    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_time_column(self, unit):
        """Dates given in a ``time_col`` column: the unit of that column."""
        flat = _small('panel', unit).reset_index()

        result = compare_and_detect_delays(flat, None, '2023-06-15', time_col='date', panel_cols=['country'])

        assert datetime_units(result) == _date_units(unit)


class TestEmptyResult:
    """An empty result has the dtypes (dates and strings) and index levels of a non-empty one."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('structure', ['series', 'panel'])
    @pytest.mark.parametrize('tz', [None, 'Europe/Paris'], ids=['naive', 'aware'])
    @pytest.mark.parametrize('download_date', [
        MICROSECOND_DOWNLOAD, pd.Timestamp('2024-10-15 08:00', tz='Asia/Tokyo'),
    ], ids=['microseconds', 'aware'])
    @pytest.mark.parametrize('emptiness', ['no_observation', 'no_change'])
    def test_empty_result_has_the_dtypes_of_a_non_empty_one(self, emptiness, download_date, tz, structure, unit):
        """No observation at all, or no change since a snapshot stored in nanoseconds: same dtypes."""
        data = _small(structure, unit, tz=tz)
        non_empty = compare_and_detect_delays(data, None, download_date)

        if emptiness == 'no_observation':
            result = compare_and_detect_delays(data * np.nan, None, download_date)
        else:
            result = compare_and_detect_delays(data, with_datetime_resolution(data, 'ns'), download_date)

        assert result.empty and list(result.columns) == OUTPUT_COLUMNS
        pd.testing.assert_series_equal(result.dtypes, non_empty.dtypes)

    @pytest.mark.parametrize('structure', ['series', 'panel'])
    def test_empty_result_strings_have_the_dtypes_of_a_non_empty_one(self, structure):
        """The string columns and index levels of an empty result are typed like a non-empty one."""
        data = _small(structure, 'us')
        non_empty = compare_and_detect_delays(data, None, '2024-10-15')

        result = compare_and_detect_delays(data * np.nan, None, '2024-10-15')

        pd.testing.assert_series_equal(result.dtypes, non_empty.dtypes)
        assert [level.dtype for level in getattr(result.index, 'levels', [result.index])] == [
            level.dtype for level in getattr(non_empty.index, 'levels', [non_empty.index])
        ]


class TestApplicableDelay:
    """The applicable delays derived from a result do not depend on the resolution."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('kwargs', [
        dict(reference_point='end', frequency='quarterly'),
        dict(reference_point='start', frequency='monthly', unit='hour', aggregate_by_panel=True),
    ], ids=['quarterly_end', 'monthly_start_hours'])
    def test_applicable_delays_equal_the_nanosecond_ones(self, realistic_panel, kwargs, unit):
        """Same applicable delays, to the bit, whatever the unit of the detected dates."""
        def applicable(data_unit):
            detected = compare_and_detect_delays(
                with_datetime_resolution(realistic_panel, data_unit), None, MICROSECOND_DOWNLOAD
            )
            return calculate_applicable_delay(detected, **kwargs)

        result = applicable(unit)

        pd.testing.assert_frame_equal(result, applicable('ns'), check_exact=True)
        # Aucune date dans le résultat : rien à exprimer dans une unité
        assert datetime_units(result) == {}
