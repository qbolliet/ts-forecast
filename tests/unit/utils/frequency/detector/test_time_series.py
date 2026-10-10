"""Tests for ``FrequencyDetector.detect_time_series_frequency`` (``utils/frequency/detector.py``).

Covers the detection on a single series, through ``pd.infer_freq`` and the heuristic
fallback (``_extend_infer_freq`` and its helpers, reached through the public method):
every frequency (``D``, ``B``, ``W``, ``SM``, ``M``, ``Q``, ``Y``, intraday, sub-second)
with its start / end position in every ``return_format``; two observations; gaps;
non-default anchors (ANO-UTILS-043); multiplied frequencies and the repeated-majority
rule of the fallback (ANO-UTILS-051); too few observations (``min_observations``);
unsorted and duplicated dates (ANO-UTILS-014); index types (strings, time zones,
labels, integers ANO-UTILS-044, ``Period`` ANO-UTILS-045).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import shuffle_rows
from tests.unit.utils.frequency.detector.helpers import RETURN_FORMATS, is_on_grid, make_series
from tsforecast.utils.frequency.detector import FrequencyDetector
from tsforecast.utils.frequency.utils import detect_index_frequency


# =============================================================================
# FrequencyDetector.detect_time_series_frequency : fréquences et positions
# =============================================================================

# Grilles régulières (au moins 3 dates : chemin pd.infer_freq) et valeur d'or de chaque
# format. Valeurs d'or : code de base sans position ; position S / E des fréquences
# période ; chaîne pandas complète avec ancre canonique ('QS-OCT' -> 'QS-JAN', cf.
# canonicalize_frequency) ; composants (base, position, ancre, multiplicateur)
REGULAR_GRIDS = [
    ('D', '2024-01-01', ('D', 'D', 'D', ('D', None, None, 1))),
    ('B', '2024-01-01', ('B', 'B', 'B', ('B', None, None, 1))),
    ('W', '2024-01-07', ('W', 'W', 'W-SUN', ('W', None, 'SUN', 1))),
    ('W-MON', '2024-01-01', ('W', 'W', 'W-MON', ('W', None, 'MON', 1))),
    ('MS', '2024-01-01', ('M', 'MS', 'MS', ('M', 'S', None, 1))),
    ('ME', '2024-01-31', ('M', 'ME', 'ME', ('M', 'E', None, 1))),
    ('QS', '2024-01-01', ('Q', 'QS', 'QS-JAN', ('Q', 'S', 'JAN', 1))),
    ('QE', '2024-03-31', ('Q', 'QE', 'QE-DEC', ('Q', 'E', 'DEC', 1))),
    ('YS', '2020-01-01', ('Y', 'YS', 'YS-JAN', ('Y', 'S', 'JAN', 1))),
    ('YE', '2020-12-31', ('Y', 'YE', 'YE-DEC', ('Y', 'E', 'DEC', 1))),
    ('h', '2024-01-01', ('h', 'h', 'h', ('h', None, None, 1))),
    ('min', '2024-01-01', ('min', 'min', 'min', ('min', None, None, 1))),
    ('s', '2024-01-01', ('s', 's', 's', ('s', None, None, 1))),
]

# Grilles dont l'ancre est celle par défaut de pandas : le code sans ancre
# ('W', 'QS', 'YE', ...) décrit exactement les mêmes dates
DEFAULT_ANCHOR_GRIDS = [
    ('D', '2024-01-01'), ('W', '2024-01-07'), ('MS', '2024-01-01'), ('ME', '2024-01-31'),
    ('QS', '2024-01-01'), ('QE', '2024-03-31'), ('YS', '2010-01-01'), ('YE', '2010-12-31'),
]

# Grilles à ancre non par défaut : semaine au lundi, trimestres décalés, exercice en juillet
NON_DEFAULT_ANCHOR_GRIDS = [
    ('W-MON', '2024-01-01'), ('QS-FEB', '2020-02-01'), ('QE-NOV', '2020-11-30'),
    ('YS-JUL', '2010-07-01'), ('YE-JUN', '2010-06-30'),
]


class TestDetectTimeSeriesFrequencyRegularGrids:
    """Every frequency and position is detected exactly, in every ``return_format``."""

    @pytest.mark.parametrize('return_format', RETURN_FORMATS)
    @pytest.mark.parametrize(
        'freq, start, expected', [pytest.param(*grid, id=grid[0]) for grid in REGULAR_GRIDS]
    )
    def test_regular_grid(self, freq, start, expected, return_format):
        """Six regular dates give the golden value of the requested format."""
        series = make_series(pd.date_range(start, periods=6, freq=freq))
        detected = FrequencyDetector().detect_time_series_frequency(series, return_format)
        assert detected == expected[RETURN_FORMATS.index(return_format)]

    def test_default_format_is_base(self):
        """Without ``return_format``, the base code is returned."""
        series = make_series(pd.date_range('2023-01-01', periods=10, freq='MS'))
        assert FrequencyDetector().detect_time_series_frequency(series) == 'M'

    def test_nan_values_are_ignored(self):
        """Missing values inside a daily series do not break the daily spacing."""
        # Deux NaN isolés : les dates restantes gardent un écart modal d'un jour
        values = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        series = pd.Series(values, index=pd.date_range('2023-01-01', periods=10, freq='D'))
        assert FrequencyDetector().detect_time_series_frequency(series) == 'D'

    def test_multiplied_frequency_keeps_multiplier_in_full_format(self):
        """A ten-day grid is ``'D'`` in the base format and ``'10D'`` in the full format."""
        # Le format 'base' écarte le multiplicateur par construction (normalize_frequency)
        series = make_series(pd.date_range('2024-01-01', periods=5, freq='10D'))
        detector = FrequencyDetector()
        assert (detector.detect_time_series_frequency(series, 'base'),
                detector.detect_time_series_frequency(series, 'full')) == ('D', '10D')

    def test_invalid_return_format_raises(self):
        """An unknown ``return_format`` is rejected rather than silently ignored."""
        series = make_series(pd.date_range('2024-01-01', periods=5, freq='D'))
        with pytest.raises(ValueError, match='Invalid return_format'):
            FrequencyDetector().detect_time_series_frequency(series, 'bogus')


class TestDetectTimeSeriesFrequencyTwoObservations:
    """Two dates are enough: the heuristic fallback takes over from ``pd.infer_freq``."""

    # Valeurs d'or : écart modal unique -> code de base ; position lue sur les deux dates
    @pytest.mark.parametrize(
        'freq, start, expected',
        [
            pytest.param('D', '2024-01-01', ('D', 'D'), id='D'),
            pytest.param('W', '2024-01-07', ('W', 'W'), id='W'),
            pytest.param('MS', '2024-01-01', ('M', 'MS'), id='MS'),
            pytest.param('ME', '2024-01-31', ('M', 'ME'), id='ME'),
            pytest.param('QS', '2024-01-01', ('Q', 'QS'), id='QS'),
            pytest.param('QE', '2024-03-31', ('Q', 'QE'), id='QE'),
            pytest.param('YS', '2020-01-01', ('Y', 'YS'), id='YS'),
            pytest.param('YE', '2020-12-31', ('Y', 'YE'), id='YE'),
            pytest.param('h', '2024-01-01', ('h', 'h'), id='h'),
            pytest.param('min', '2024-01-01', ('min', 'min'), id='min'),
            pytest.param('s', '2024-01-01', ('s', 's'), id='s'),
        ],
    )
    def test_base_and_position(self, freq, start, expected):
        """Base code and position are both recovered from two dates."""
        series = make_series(pd.date_range(start, periods=2, freq=freq))
        detector = FrequencyDetector()
        detected = (detector.detect_time_series_frequency(series, 'base'),
                    detector.detect_time_series_frequency(series, 'with_position'))
        assert detected == expected

    @pytest.mark.parametrize(
        'freq, start', [pytest.param(*grid, id=grid[0]) for grid in DEFAULT_ANCHOR_GRIDS]
    )
    def test_full_format_regenerates_the_dates(self, freq, start):
        """Property: the full detected offset regenerates both dates."""
        dates = pd.date_range(start, periods=2, freq=freq)
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full')
        assert list(pd.date_range(dates[0], periods=2, freq=detected)) == list(dates)

    @pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
    def test_sub_second_unit_spacing(self, unit):
        """A spacing of exactly one sub-second unit is that unit."""
        series = make_series(pd.date_range('2024-01-01', periods=2, freq=unit))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == unit

    @pytest.mark.parametrize('spacing', ['10ms', '10us', '10ns'])
    def test_sub_second_multiple_on_gapped_grid(self, spacing):
        """A repeated sub-second spacing keeps its multiplier (no longer rounded to its unit)."""
        dates = pd.date_range('2024-01-01', periods=6, freq=spacing).delete([3])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == spacing

    def test_consecutive_business_days_without_weekend_are_daily(self):
        """Monday and Tuesday alone cannot be told apart from calendar days: ``'D'``."""
        # Aucun week-end dans l'intervalle : rien ne distingue 'B' de 'D'
        series = make_series(pd.DatetimeIndex(['2024-01-01', '2024-01-02']))
        assert FrequencyDetector().detect_time_series_frequency(series) == 'D'

    @pytest.mark.parametrize(
        'dates',
        [
            # Vendredi puis lundi : écart unique de 3 jours
            pytest.param(['2024-01-05', '2024-01-08'], id='friday-monday'),
            # Écart de 10 jours : ni hebdomadaire, ni semi-mensuel
            pytest.param(['2024-01-01', '2024-01-11'], id='ten-days'),
            # Écart de 90 minutes : ni horaire, ni minute
            pytest.param(['2024-01-01 00:00', '2024-01-01 01:30'], id='ninety-minutes'),
            # Écart de 14 jours entre deux dimanches (7 et 21 du mois : pas semi-mensuel)
            pytest.param(['2024-01-07', '2024-01-21'], id='fourteen-days-sundays'),
            # Écart de 2 mois entre deux débuts de mois
            pytest.param(['2024-01-01', '2024-03-01'], id='two-months'),
        ],
    )
    def test_single_spacing_never_gives_a_multiplied_frequency(self, dates):
        """A spacing seen once is no evidence of a multiplied frequency: ``None``."""
        # Deux dates quelconques ont toujours un écart : un multiplicateur exige que
        # l'écart se répète (comme pd.infer_freq, qui exige trois dates)
        series = make_series(pd.DatetimeIndex(dates))
        assert FrequencyDetector().detect_time_series_frequency(series) is None

    @pytest.mark.parametrize(
        'dates, expected',
        [
            # Écarts de 3, 3 puis 6 jours (une date manquante)
            pytest.param(['2024-01-05', '2024-01-08', '2024-01-11', '2024-01-17'], '3D', id='three-days'),
            # Écarts de 2, 2 puis 4 mois entre débuts de mois
            pytest.param(['2024-01-01', '2024-03-01', '2024-05-01', '2024-09-01'], '2MS', id='two-months'),
        ],
    )
    def test_repeated_spacing_gives_a_multiplied_frequency(self, dates, expected):
        """Three dates sharing a repeated, majority spacing are enough for a multiplier."""
        series = make_series(pd.DatetimeIndex(dates))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == expected


class TestDetectTimeSeriesFrequencyWithGaps:
    """Missing dates do not change the detected frequency (modal spacing)."""

    # Suppression des 4e, 8e et 9e dates d'une grille de 12 : pd.infer_freq échoue,
    # l'écart modal reste celui de la grille
    GAPS = [3, 7, 8]

    @pytest.mark.parametrize(
        'freq, start, expected',
        [
            pytest.param('D', '2024-01-01', 'D', id='D'),
            pytest.param('B', '2024-01-01', 'B', id='B'),
            pytest.param('h', '2024-01-01', 'h', id='h'),
            pytest.param('W', '2024-01-07', 'W', id='W'),
            pytest.param('MS', '2024-01-01', 'MS', id='MS'),
            pytest.param('ME', '2024-01-31', 'ME', id='ME'),
            pytest.param('QS', '2020-01-01', 'QS', id='QS'),
            pytest.param('QE', '2020-03-31', 'QE', id='QE'),
            pytest.param('YS', '2010-01-01', 'YS', id='YS'),
            pytest.param('YE', '2010-12-31', 'YE', id='YE'),
        ],
    )
    def test_frequency_with_position(self, freq, start, expected):
        """Gapped grids keep their frequency and position."""
        dates = pd.date_range(start, periods=12, freq=freq).delete(self.GAPS)
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'with_position')
        assert detected == expected

    @pytest.mark.parametrize(
        'freq, start', [pytest.param(*grid, id=grid[0]) for grid in DEFAULT_ANCHOR_GRIDS]
    )
    def test_full_format_lies_on_the_data_grid(self, freq, start):
        """Property: every observed date lies on the full detected offset."""
        dates = pd.date_range(start, periods=12, freq=freq).delete(self.GAPS)
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full')
        assert is_on_grid(dates, detected)

    @pytest.mark.parametrize(
        'freq, start',
        [pytest.param('D', '2024-03-20', id='daily'), pytest.param('MS', '2024-01-01', id='month-start')],
    )
    def test_timezone_aware_gapped_grid(self, freq, start):
        """Daylight saving time (23 h / 25 h days) does not blur a gapped grid."""
        # Grille à cheval sur le passage à l'heure d'été (31 mars 2024 à Paris)
        dates = pd.date_range(start, periods=12, freq=freq, tz='Europe/Paris').delete(self.GAPS)
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == freq


class TestFallbackMultipliedFrequencies:
    """The fallback reports multiplied frequencies, as ``pd.infer_freq`` does on regular grids."""

    @pytest.mark.parametrize(
        'freq, start',
        [
            pytest.param('2MS', '2024-01-01', id='two-months'),
            # Semestres : pandas les écrit '2QS-OCT', forme canonique '2QS-JAN'
            pytest.param('6MS', '2020-01-01', id='half-years'),
            pytest.param('2QE-NOV', '2020-11-30', id='two-quarters-november'),
            pytest.param('2YS', '2000-01-01', id='two-years'),
            pytest.param('2W-WED', '2024-01-03', id='two-weeks'),
            pytest.param('3D', '2024-01-01', id='three-days'),
            pytest.param('90min', '2024-01-01', id='ninety-minutes'),
            pytest.param('36h', '2024-01-01', id='thirty-six-hours'),
        ],
    )
    def test_gapped_grid_matches_regular_grid(self, freq, start):
        """Property: a gapped grid is detected as the regular grid it comes from."""
        regular = pd.date_range(start, periods=12, freq=freq)
        detector = FrequencyDetector()
        expected = detector.detect_time_series_frequency(make_series(regular), 'full')
        gapped = regular.delete([3, 7, 8])
        assert detector.detect_time_series_frequency(make_series(gapped), 'full') == expected

    @pytest.mark.parametrize(
        'return_format, expected',
        [
            pytest.param('base', 'W', id='base'),
            pytest.param('with_position', 'W', id='with-position'),
            pytest.param('full', '2W-WED', id='full'),
            pytest.param('components', ('W', None, 'WED', 2), id='components'),
        ],
    )
    def test_multiplier_by_format(self, return_format, expected):
        """Only 'full' and 'components' carry the multiplier."""
        dates = pd.date_range('2024-01-03', periods=8, freq='2W-WED').delete([3])
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), return_format)
        assert detected == expected

    @pytest.mark.parametrize(
        'dates',
        [
            # Écarts de 45 puis 50 jours : aucun écart majoritaire
            pytest.param(['2023-01-01', '2023-02-15', '2023-04-06'], id='two-different-spacings'),
            # Écarts de 10, 12, 10, 12 jours : 10 jours modal mais pas majoritaire
            pytest.param(['2024-01-01', '2024-01-11', '2024-01-23', '2024-02-02', '2024-02-14'],
                         id='tie-between-spacings'),
            # Débuts de mois espacés de 2, 3, puis 5 mois : 2 mois modal, minoritaire
            pytest.param(['2024-01-01', '2024-03-01', '2024-06-01', '2024-11-01'],
                         id='calendar-tie'),
            # Mercredis espacés de 2, 3, 2, 3 semaines : 2 semaines modal, pas majoritaire
            pytest.param(['2024-01-03', '2024-01-17', '2024-02-07', '2024-02-21', '2024-03-13'],
                         id='weekly-tie'),
            # Écarts de 90, 100, 90, 100 minutes
            pytest.param(['2024-01-01 00:00', '2024-01-01 01:30', '2024-01-01 03:10',
                          '2024-01-01 04:40', '2024-01-01 06:20'], id='intraday-tie'),
        ],
    )
    def test_minority_multiplied_spacing_is_undetectable(self, dates):
        """A multiplied spacing shared by at most half of the spacings gives ``None``."""
        series = make_series(pd.DatetimeIndex(dates))
        assert FrequencyDetector().detect_time_series_frequency(series) is None

    @pytest.mark.parametrize(
        'seconds, expected',
        [pytest.param(3601, 'h', id='hour-plus-one-second'),
         pytest.param(59, 'min', id='minute-minus-one-second')],
    )
    def test_jitter_around_a_unit_is_tolerated(self, seconds, expected):
        """A spacing within 5 % of one hour or one minute is read as that unit."""
        dates = pd.date_range('2024-01-01', periods=2, freq=f'{seconds}s')
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == expected


class TestFallbackWithoutCalendarPattern:
    """Period frequencies are still read on dates whose day of the month varies."""

    @pytest.mark.parametrize(
        'dates, expected',
        [
            # Jours 3, 2, 3, 5 du mois : écart modal de 30 jours
            pytest.param(['2024-01-03', '2024-02-02', '2024-03-03', '2024-04-05'], 'M', id='monthly'),
            # Écart modal de 91 jours (13 semaines), jours du mois et de la semaine variables
            pytest.param(['2024-01-03', '2024-04-03', '2024-07-03', '2024-10-05'], 'Q', id='quarterly'),
            # Écart modal de 365 jours, jours du mois variables
            pytest.param(['2021-01-03', '2022-01-03', '2023-01-03', '2024-01-06'], 'Y', id='annual'),
        ],
    )
    def test_period_without_position(self, dates, expected):
        """Neither position nor anchor can be read: the bare period code is reported."""
        series = make_series(pd.DatetimeIndex(dates))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == expected

    def test_multiple_of_seven_days_between_varying_weekdays_is_not_weekly(self):
        """A two-week modal spacing between different weekdays is counted in days."""
        # Écarts de 14, 14, 15 jours : lundis puis un mardi
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-15', '2024-01-29', '2024-02-13'])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == '14D'

    def test_unsupported_pandas_alias_falls_back(self):
        """A business month end grid (``'BME'`` for pandas, not supported here) is monthly."""
        dates = pd.date_range('2024-01-31', periods=6, freq='BME')
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates)) == 'M'


class TestFallbackKeepsNonDefaultAnchors:
    """The heuristic fallback reports an offset whose grid holds the observed dates (ANO-UTILS-043)."""

    @pytest.mark.parametrize(
        'freq, start', [pytest.param(*grid, id=grid[0]) for grid in NON_DEFAULT_ANCHOR_GRIDS]
    )
    def test_gapped_grid(self, freq, start):
        """A gapped grid anchored on Monday / February / July is detected on that anchor."""
        dates = pd.date_range(start, periods=12, freq=freq).delete([3, 7, 8])
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full')
        assert is_on_grid(dates, detected)

    @pytest.mark.parametrize(
        'freq, start', [pytest.param(*grid, id=grid[0]) for grid in NON_DEFAULT_ANCHOR_GRIDS]
    )
    def test_two_observations(self, freq, start):
        """Two dates anchored on Monday / February / July are detected on that anchor."""
        dates = pd.date_range(start, periods=2, freq=freq)
        detected = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full')
        assert is_on_grid(dates, detected)

    @pytest.mark.parametrize(
        'freq, start, expected',
        [
            pytest.param('W-MON', '2024-01-01', 'W-MON', id='W-MON'),
            pytest.param('QS-FEB', '2020-02-01', 'QS-FEB', id='QS-FEB'),
            # Trimestres terminés en nov., févr., mai, août : fin canonique en février
            pytest.param('QE-NOV', '2020-11-30', 'QE-FEB', id='QE-NOV'),
            pytest.param('YS-JUL', '2010-07-01', 'YS-JUL', id='YS-JUL'),
            pytest.param('YE-JUN', '2010-06-30', 'YE-JUN', id='YE-JUN'),
        ],
    )
    def test_gapped_grid_gives_the_canonical_anchor(self, freq, start, expected):
        """Golden anchor: the one ``pd.infer_freq`` reports on the regular grid, in canonical form."""
        dates = pd.date_range(start, periods=12, freq=freq).delete([3, 7, 8])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == expected


# =============================================================================
# FrequencyDetector.detect_time_series_frequency : cas limites
# =============================================================================

class TestDetectTimeSeriesFrequencyTooFewObservations:
    """Fewer than ``min_observations`` non-null values raise a ``ValueError``."""

    @pytest.mark.parametrize(
        'values, dates, count',
        [
            pytest.param([1.0], ['2024-01-01'], 1, id='single-observation'),
            pytest.param([], [], 0, id='empty'),
            pytest.param([np.nan] * 3, ['2024-01-01', '2024-01-02', '2024-01-03'], 0, id='all-nan'),
            # Une seule valeur observée parmi trois dates
            pytest.param([np.nan, 1.0, np.nan], ['2024-01-01', '2024-01-02', '2024-01-03'], 1,
                         id='single-non-null'),
        ],
    )
    def test_raises(self, values, dates, count):
        """The error states the number of non-null observations."""
        series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)
        with pytest.raises(ValueError, match=f'only {count} non-null observations'):
            FrequencyDetector(min_observations=2).detect_time_series_frequency(series)

    def test_custom_minimum(self):
        """``min_observations=3`` rejects a two-date series that the default accepts."""
        series = make_series(pd.date_range('2024-01-01', periods=2, freq='MS'))
        with pytest.raises(ValueError, match='minimum required is 3'):
            FrequencyDetector(min_observations=3).detect_time_series_frequency(series)

    def test_single_observation_accepted_but_undetectable(self):
        """With ``min_observations=1``, one date is accepted but carries no spacing: ``None``."""
        series = make_series(pd.DatetimeIndex(['2024-01-01']))
        assert FrequencyDetector(min_observations=1).detect_time_series_frequency(series) is None


class TestDetectTimeSeriesFrequencyUnsortedData:
    """The order of the rows never changes the detected frequency."""

    @pytest.mark.parametrize(
        'freq, start, expected',
        [
            pytest.param('D', '2023-01-01', 'D', id='D'),
            pytest.param('MS', '2023-01-01', 'MS', id='MS'),
            pytest.param('QE', '2023-03-31', 'QE-DEC', id='QE'),
        ],
    )
    def test_shuffled_series(self, freq, start, expected):
        """A shuffled regular series gives the frequency of the sorted one."""
        series = shuffle_rows(make_series(pd.date_range(start, periods=10, freq=freq)), seed=3)
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == expected

    def test_unsorted_with_duplicates(self):
        """Duplicated and unsorted monthly dates are still monthly, start anchored."""
        # Trois dates uniques (janvier, février, mars, avril), deux doublons, ordre mélangé
        dates = pd.DatetimeIndex(['2024-03-01', '2024-01-01', '2024-02-01',
                                  '2024-01-01', '2024-04-01', '2024-02-01'])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == 'MS'

    def test_reversed_series(self):
        """A series in decreasing order is detected as the sorted one."""
        series = make_series(pd.date_range('2023-01-01', periods=10, freq='MS'))[::-1]
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == 'MS'


class TestFrequencyDetectorDuplicatedDates:
    """Duplicated dates never yield a zero spacing (ANO-UTILS-014)."""

    def test_short_index_with_duplicate_is_monthly(self):
        """A duplicated date among few monthly dates does not turn the frequency into 'ns'.

        Écarts triés avec doublon : 0, 31, 29 jours, tous modaux ; avant
        correction, l'écart nul l'emportait et la fréquence valait ``'ns'``.
        Après dédoublonnage, trois dates mensuelles uniques -> ``'MS'``.
        """
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-01', '2024-02-01', '2024-03-01'])
        series = pd.Series(range(4), index=dates)
        assert FrequencyDetector().detect_time_series_frequency(series, return_format='full') == 'MS'

    def test_only_duplicates_is_undetectable(self):
        """Two identical dates carry no spacing at all: no frequency."""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-01'])
        series = pd.Series(range(2), index=dates)
        assert FrequencyDetector().detect_time_series_frequency(series) is None


class TestFrequencyDetectorSemiMonthly:
    """Semi-monthly grids are detected with their position ('SMS' / 'SME')."""

    @pytest.mark.parametrize(
        "freq, expected",
        [
            pytest.param('SMS', 'SMS', id="start-grid-1st-and-15th"),
            pytest.param('SME', 'SME', id="end-grid-15th-and-month-end"),
        ],
    )
    def test_native_grids(self, freq, expected):
        """Native pandas semi-monthly grids keep their position in the 'full' format."""
        dates = pd.date_range('2024-01-01', periods=8, freq=freq)
        assert detect_index_frequency(dates, return_format='full') == expected

    def test_loose_pattern_has_no_position(self):
        """Dates around the start and middle of months (not exactly 1st / 15th) stay 'SM'."""
        # Autant de dates en début (2) qu'en milieu (16) de mois : > 40 % chacune
        dates = pd.DatetimeIndex(
            ['2024-01-02', '2024-01-16', '2024-02-02', '2024-02-16', '2024-03-02', '2024-03-16']
        )
        assert detect_index_frequency(dates, return_format='full') == 'SM'


class TestDetectTimeSeriesFrequencyIndexTypes:
    """Index types other than a plain ``DatetimeIndex``."""

    def test_string_dates_are_converted(self):
        """ISO date strings are parsed before detection."""
        series = make_series(pd.Index(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == 'MS'

    @pytest.mark.parametrize(
        'freq, expected',
        [pytest.param('D', 'D', id='daily'), pytest.param('MS', 'MS', id='month-start')],
    )
    def test_timezone_aware_index(self, freq, expected):
        """A time zone does not alter the detection."""
        dates = pd.date_range('2024-01-01', periods=4, freq=freq, tz='Europe/Paris')
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == expected

    def test_timezone_aware_two_observations(self):
        """The fallback also reads the position on a time zone aware index."""
        dates = pd.date_range('2024-01-01', periods=2, freq='MS', tz='Europe/Paris')
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == 'MS'

    # Avertissement de pd.to_datetime avant son échec : sans objet pour le test
    @pytest.mark.filterwarnings('ignore:Could not infer format')
    def test_non_date_labels_raise(self):
        """Labels that are not dates raise a ``ValueError``."""
        series = make_series(pd.Index(['a', 'b', 'c', 'd']))
        with pytest.raises(ValueError, match='cannot be converted to datetime'):
            FrequencyDetector().detect_time_series_frequency(series)

    @pytest.mark.parametrize(
        'index',
        [pytest.param(pd.Index([2020, 2021, 2022, 2023]), id='years'),
         pytest.param(pd.RangeIndex(4), id='range-index'),
         pytest.param(pd.Index([0.5, 1.5, 2.5]), id='floats')],
    )
    def test_numeric_index_is_rejected(self, index):
        """A numeric index is not a time index (ANO-UTILS-044, ANO-UTILS-028 decision)."""
        with pytest.raises(ValueError, match='numeric labels are not dates'):
            FrequencyDetector().detect_time_series_frequency(make_series(index))

    @pytest.mark.parametrize(
        'freq, expected',
        [pytest.param('M', 'MS', id='monthly'),
         pytest.param('Q', 'QS-JAN', id='quarterly'),
         pytest.param('Y', 'YS-JAN', id='annual')],
    )
    def test_period_index_is_read_at_period_start(self, freq, expected):
        """Periods are read at their first instant (ANO-UTILS-045, ANO-UTILS-029 decision)."""
        series = make_series(pd.period_range('2024-01', periods=4, freq=freq))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == expected


class TestFallbackWithOutlierDates:
    """A 75 % of dates at month starts / ends keeps the position of the grid."""

    @staticmethod
    def _monthly_grid_with_outlier(freq: str, start: str, outlier: str) -> pd.DatetimeIndex:
        """Build a ten-month grid whose sixth date is replaced by an off-grid ``outlier``."""
        dates = list(pd.date_range(start, periods=10, freq=freq))
        dates[5] = pd.Timestamp(outlier)
        return pd.DatetimeIndex(dates)

    @pytest.mark.parametrize(
        'freq, start, outlier, expected',
        [
            pytest.param('MS', '2024-01-01', '2024-06-12', 'MS', id='month-start'),
            pytest.param('ME', '2024-01-31', '2024-06-12', 'ME', id='month-end'),
        ],
    )
    def test_outlier_does_not_hide_the_position(self, freq, start, outlier, expected):
        """Nine dates on the grid and one stray date: the grid is read, with its position."""
        dates = self._monthly_grid_with_outlier(freq, start, outlier)
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == expected

    def test_quarterly_outlier_keeps_position_and_anchor(self):
        """Quarter starts with a stray date: position 'S' and anchor of the aligned dates."""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-15', '2025-01-01'])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == 'QS-JAN'

    def test_annual_outlier_keeps_position_and_anchor(self):
        """Year starts with a stray date: position 'S' and anchor January."""
        dates = pd.DatetimeIndex(['2020-01-01', '2021-01-01', '2022-03-15', '2023-01-01', '2024-01-01'])
        assert FrequencyDetector().detect_time_series_frequency(make_series(dates), 'full') == 'YS-JAN'

    def test_components_expose_position_and_suffix(self):
        """The rich ``components`` format carries the recovered position and anchor."""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-15', '2025-01-01'])
        parsed = FrequencyDetector().detect_time_series_frequency(make_series(dates), 'components')
        assert (parsed.freq, parsed.position, parsed.suffix) == ('Q', 'S', 'JAN')

    @pytest.mark.parametrize(
        'dates, expected',
        [
            pytest.param(['2024-01-01', '2024-02-01', '2024-03-10', '2024-04-20'], 'M', id='half'),
            pytest.param(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-15', '2023-05-30'], None,
                         id='three-fifths'),
        ],
    )
    def test_fewer_than_three_quarters_aligned_keeps_no_position(self, dates, expected):
        """Below 75 % of month starts the outliers are too many: no position is read."""
        series = make_series(pd.DatetimeIndex(dates))
        assert FrequencyDetector().detect_time_series_frequency(series, 'full') == expected
