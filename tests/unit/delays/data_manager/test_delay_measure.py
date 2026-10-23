"""Delay measurement of ``compare_and_detect_delays``.

Scenarios: reference point and unit, accepted forms of ``download_date``, per-column
frequency and period bounds, and the delay unit resolution.
"""

# Modules de base
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import (
    TS,
    _expected,
    _months,
    _panel,
    _records,
    _series,
)


# Le chemin « colonnes » (time_col / panel_cols) émet systématiquement un
# avertissement de remplacement d'index (comportement de validate_temporal_data) :
# le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")


# =============================================================================
# Point de référence, unité et arrondi
# =============================================================================
class TestReferencePointAndUnit:
    """Delay measured from the start or the end of the period, in days, seconds or microseconds."""

    @pytest.mark.parametrize(
        ('reference_point', 'expected_delay'),
        [('start', 75.0), ('end', 45.0)],
        ids=['start', 'end'],
    )
    def test_reference_point(self, monthly_series, reference_point, expected_delay):
        """The delay is counted from the period start or from the (exclusive) period end."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15', reference_point=reference_point)

        # Valeurs d'or : 1er avril -> 15 juin = 75 j ; 1er mai -> 15 juin = 45 j
        assert result[['reference_point', 'delay']].iloc[0].tolist() == [reference_point, expected_delay]

    def test_reference_point_defaults_to_start(self, monthly_series):
        """Without ``reference_point`` the delay is counted from the period start."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert result['reference_point'].iloc[0] == 'start'

    def test_invalid_reference_point_raises(self, monthly_series):
        """Only ``'start'`` and ``'end'`` are accepted."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            compare_and_detect_delays(monthly_series, None, '2023-06-15', reference_point='middle')

    @pytest.mark.parametrize(
        ('delay_unit', 'expected_unit', 'expected_delay'),
        [
            ('D', 'day', 75.0), ('day', 'day', 75.0),
            ('s', 'second', 6_480_000.0), ('second', 'second', 6_480_000.0),
            ('us', 'microsecond', 6_480_000_000_000.0), ('microsecond', 'microsecond', 6_480_000_000_000.0),
        ],
        ids=['D', 'day', 's', 'second', 'us', 'microsecond'],
    )
    def test_delay_unit(self, monthly_series, delay_unit, expected_unit, expected_delay):
        """Short and long unit names are equivalent; the ``unit`` column carries the long name."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15', delay_unit=delay_unit)

        # Valeur d'or : 75 j = 75 * 86 400 s = 6 480 000 s = 6,48e12 µs
        assert result[['unit', 'delay']].iloc[0].tolist() == [expected_unit, expected_delay]

    def test_delay_unit_defaults_to_day(self, monthly_series):
        """Without ``delay_unit`` the delay is in days."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert result['unit'].iloc[0] == 'day'

    @pytest.mark.parametrize('delay_unit', ['hour', 'invalid', 'Day', 'ms'])
    def test_invalid_delay_unit_raises(self, monthly_series, delay_unit):
        """Units other than day / second / microsecond are rejected."""
        with pytest.raises(ValueError, match="Unit must be one of"):
            compare_and_detect_delays(monthly_series, None, '2023-06-15', delay_unit=delay_unit)

    @pytest.mark.parametrize(
        ('download_date', 'delay_unit', 'expected_delay'),
        [
            ('2023-06-15 12:00', 'day', 76.0),
            ('2023-06-15 00:00:00.5', 'second', 6_480_001.0),
            ('2023-06-15 00:00:00.000001', 'second', 6_480_001.0),
            ('2023-06-15 00:00:00.000001', 'day', 76.0),
        ],
        ids=['half-day', 'half-second', 'one-microsecond-in-seconds', 'one-microsecond-in-days'],
    )
    def test_delay_is_rounded_up(self, monthly_series, download_date, delay_unit, expected_delay):
        """Any fraction of the unit rounds the delay up to the next integer."""
        result = compare_and_detect_delays(monthly_series, None, download_date, delay_unit=delay_unit)

        # Valeurs d'or : 75,5 j -> 76 ; 6 480 000,5 s -> 6 480 001 ; 75 j + 1 µs -> 76 j
        assert result['delay'].iloc[0] == expected_delay

    def test_exact_whole_unit_is_not_rounded_up(self, monthly_series):
        """A delay of exactly 75 days stays 75 (no spurious ``+ 1``)."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15 00:00:00')

        assert result['delay'].iloc[0] == 75.0

    def test_download_before_the_period_start_gives_a_negative_delay(self, monthly_series):
        """A download dated before the observed period yields a negative delay, rounded towards ``+inf``."""
        result = compare_and_detect_delays(monthly_series, None, '2023-03-20 12:00')

        # Valeur d'or : 20 mars 12:00 -> 1er avril = 11,5 j avant, soit -11,5 arrondi à -11.
        # Le signe est conservé (cohérent avec calculate_applicable_delay), sans erreur ni avertissement.
        assert result['delay'].iloc[0] == -11.0

    @pytest.mark.parametrize(
        'download_date',
        [
            datetime(2027, 11, 26, 17, 18, 56, 578903),
            datetime(2027, 3, 25, 11, 39, 36, 422959),
            datetime(2026, 12, 26, 8, 35, 13, 531343),
        ],
        ids=['2027-11', '2027-03', '2026-12'],
    )
    def test_microsecond_delay_is_exact(self, monthly_series, download_date):
        """The microsecond delay equals the exact integer count of microseconds elapsed."""
        result = compare_and_detect_delays(monthly_series, None, download_date, delay_unit='us')

        # Valeur d'or par arithmétique entière : (téléchargement - 1er avril 2023) // 1 µs
        exact = (download_date - datetime(2023, 4, 1)) // timedelta(microseconds=1)
        assert int(result['delay'].iloc[0]) == exact

    def test_microsecond_delay_is_exact_over_decades(self):
        """Thirty-three years of delay are still counted to the microsecond."""
        data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0]}, index=pd.date_range('1990-01-01', periods=3, freq='MS'))
        download_date = datetime(2023, 12, 15, 12, 34, 56, 123457)

        result = compare_and_detect_delays(data, None, download_date, delay_unit='us')

        # Valeur d'or par arithmétique entière : (téléchargement - 1er mars 1990) // 1 µs
        exact = (download_date - datetime(1990, 3, 1)) // timedelta(microseconds=1)
        assert int(result['delay'].iloc[0]) == exact


# =============================================================================
# Date de téléchargement
# =============================================================================
class TestDownloadDate:
    """Accepted forms of ``download_date`` and their resolution."""

    @pytest.mark.parametrize(
        'download_date',
        ['2023-06-15', '2023-06-15 00:00:00', datetime(2023, 6, 15), TS('2023-06-15')],
        ids=['iso-date', 'iso-datetime', 'datetime', 'timestamp'],
    )
    def test_equivalent_forms_give_the_same_result(self, monthly_series, download_date):
        """A string, a ``datetime`` and a ``Timestamp`` naming the same instant are interchangeable."""
        result = compare_and_detect_delays(monthly_series, None, download_date)

        assert result[['download_date', 'delay']].iloc[0].tolist() == [TS('2023-06-15'), 75.0]

    def test_time_of_day_is_kept(self, monthly_series):
        """The time of day of a datetime is kept in the ``download_date`` column."""
        result = compare_and_detect_delays(monthly_series, None, datetime(2023, 6, 15, 12, 30, 45))

        assert result['download_date'].iloc[0] == TS('2023-06-15 12:30:45')

    def test_none_uses_the_current_time(self, monthly_series):
        """``download_date=None`` stamps the rows with the current time."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series, None, None)
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_default_is_the_current_time(self, monthly_series):
        """Omitting ``download_date`` behaves as ``None``."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series)
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_today_string_uses_the_current_time(self, monthly_series):
        """The string ``'today'`` (handled by ``resolve_date``) means the current time."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series, None, 'today')
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_timezone_aware_download_date_with_timezone_aware_data(self):
        """Aware data and an aware download date in the same zone give the same delay as naive ones."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, TS('2023-06-15', tz='Europe/Paris'))

        assert result['delay'].iloc[0] == 75.0

    @pytest.mark.parametrize(
        'download_date',
        [TS('2023-06-15', tz='UTC'), TS('2023-06-15 02:00', tz='Europe/Paris'), TS('2023-06-14 20:00', tz='America/New_York')],
        ids=['utc', 'paris', 'new-york'],
    )
    def test_timezone_aware_download_date_with_naive_data(self, monthly_series, download_date):
        """Naive data are read as UTC: an aware download date is compared as an instant."""
        result = compare_and_detect_delays(monthly_series, None, download_date)

        # Valeur d'or : 2023-06-15 00:00 UTC pour les trois dates (Paris est à UTC+2 en juin, New York à UTC-4),
        # soit 75 j après le 1er avril 00:00 lu en UTC
        assert result['delay'].iloc[0] == 75.0

    def test_naive_download_date_with_timezone_aware_data(self):
        """A naive download date is read as UTC when the data carry a time zone."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        # Valeur d'or : 1er avril 00:00 à Paris (UTC+2) = 31 mars 22:00 UTC ; 31 mars 22:00 -> 15 juin 00:00
        # = 75 j + 2 h, arrondi à 76 j
        assert result['delay'].iloc[0] == 76.0

    def test_aware_download_date_in_another_zone_than_the_data(self):
        """Aware data and an aware download date of another zone are compared as instants."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, TS('2023-06-14 18:00', tz='America/New_York'))

        # Valeur d'or : 18:00 à New York (UTC-4) = 22:00 UTC le 14 juin ; du 31 mars 22:00 UTC au 14 juin 22:00 UTC = 75 j
        assert result['delay'].iloc[0] == 75.0

    def test_download_date_column_keeps_the_given_time_zone(self, monthly_series):
        """The ``download_date`` column holds the date as given, time zone included."""
        given = TS('2023-06-15 02:00', tz='Europe/Paris')

        result = compare_and_detect_delays(monthly_series, None, given)

        assert result['download_date'].iloc[0] == given

    def test_timezone_aware_panel_with_timezone_aware_download_date(self, small_panel):
        """The comparison of instants also holds for panels."""
        data = small_panel.copy()
        data.index = data.index.set_levels(data.index.levels[1].tz_localize('UTC'), level='date')

        result = compare_and_detect_delays(data, None, TS('2023-06-15', tz='UTC'))

        assert _records(result, 'delay') == _expected(('A', 'PIB', 75.0), ('B', 'PIB', 106.0))

    @pytest.mark.parametrize('download_date', ['', 'not a date'], ids=['empty-string', 'garbage'])
    def test_unresolvable_string_raises(self, monthly_series, download_date):
        """A string that is not a date is refused."""
        with pytest.raises(ValueError):
            compare_and_detect_delays(monthly_series, None, download_date)

    def test_nat_raises(self, monthly_series):
        """``NaT`` is refused instead of producing ``NaN`` delays."""
        with pytest.raises(ValueError, match="NaT"):
            compare_and_detect_delays(monthly_series, None, pd.NaT)

    @pytest.mark.parametrize('download_date', [date(2023, 6, 15), 20230615, 1.5], ids=['date', 'int', 'float'])
    def test_other_types_raise(self, monthly_series, download_date):
        """Only strings and ``datetime`` objects are accepted (a plain ``date`` is not a ``datetime``)."""
        with pytest.raises(ValueError, match="'date' must be 'today', a string or datetime"):
            compare_and_detect_delays(monthly_series, None, download_date)


# =============================================================================
# Fréquence détectée par (entité, colonne) et bornes de période
# =============================================================================
class TestFrequencyAndPeriodBoundaries:
    """The frequency is detected per (entity, column) and drives the period bounds."""

    @pytest.mark.parametrize(
        ('index', 'download_date', 'delay_unit', 'frequency', 'period_start', 'period_end', 'delay_from_start', 'delay_from_end'),
        [
            (pd.date_range('2023-03-01', '2023-03-10', freq='D'), '2023-03-20', 'day',
             'daily', '2023-03-10', '2023-03-11', 10.0, 9.0),
            (pd.date_range('2023-02-26', periods=4, freq='W-SUN'), '2023-03-20', 'day',
             'weekly', '2023-03-13', '2023-03-20', 7.0, 0.0),
            (pd.date_range('2023-01-01', periods=4, freq='MS'), '2023-06-15', 'day',
             'monthly', '2023-04-01', '2023-05-01', 75.0, 45.0),
            (pd.date_range('2023-01-01', periods=4, freq='QS'), '2024-02-15', 'day',
             'quarterly', '2023-10-01', '2024-01-01', 137.0, 45.0),
            (pd.date_range('2020-01-01', periods=4, freq='YS'), '2024-02-15', 'day',
             'annual', '2023-01-01', '2024-01-01', 410.0, 45.0),
            (pd.date_range('2023-03-01 00:00', periods=5, freq='h'), '2023-03-01 06:30', 'second',
             'hourly', '2023-03-01 04:00', '2023-03-01 05:00', 9000.0, 5400.0),
        ],
        ids=['daily', 'weekly', 'monthly', 'quarterly', 'annual', 'hourly'],
    )
    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_frequency_and_bounds_of_the_last_period(
        self, index, download_date, delay_unit, frequency, period_start, period_end,
        delay_from_start, delay_from_end, reference_point,
    ):
        """The last observation's period is the one of the detected frequency, and the delay follows."""
        data = pd.DataFrame({'v': np.arange(len(index), dtype=float)}, index=index)

        result = compare_and_detect_delays(
            data, None, download_date, reference_point=reference_point, delay_unit=delay_unit,
        )

        expected_delay = delay_from_start if reference_point == 'start' else delay_from_end
        assert result[['frequency', 'period_start', 'period_end', 'delay']].iloc[0].tolist() == [
            frequency, TS(period_start), TS(period_end), expected_delay,
        ]

    def test_frequency_is_a_property_of_the_entity_and_column_couple(self, mixed_freq_panel_multifrequency):
        """One column observed annually, quarterly and monthly in three entities gets three frequencies."""
        result = compare_and_detect_delays(mixed_freq_panel_multifrequency, None, '2024-02-15')

        # Valeurs d'or (index de fin de mois, dernière date 2023-12-31) :
        #   mensuel   : période [1er déc. 2023, 1er janv. 2024) -> 15 fév. : 31 + 31 + 14 = 76 j
        #   trimestr. : période [1er oct. 2023, ...)            -> 15 fév. : 31 + 30 + 31 + 31 + 14 = 137 j
        #   annuel    : période [1er janv. 2023, ...)           -> 15 fév. : 365 + 31 + 14 = 410 j
        assert _records(result, 'frequency', 'period_start', 'delay') == _expected(
            ('DE', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('FR', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('IT', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('DE', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('FR', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('IT', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('FR', 'v', 'annual', TS('2023-01-01'), 410.0),
            ('DE', 'v', 'quarterly', TS('2023-10-01'), 137.0),
            ('IT', 'v', 'monthly', TS('2023-12-01'), 76.0),
        )

    def test_one_column_per_frequency_in_a_time_series(self):
        """Columns of different frequencies in one frame each get their own period bounds."""
        data = pd.DataFrame(
            {
                'mensuelle': np.arange(12, dtype=float),
                'trimestrielle': [1.0, np.nan, np.nan, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, 4.0, np.nan, np.nan],
            },
            index=_months(12),
        )

        result = compare_and_detect_delays(data, None, '2024-02-15')

        # Valeurs d'or : mensuelle -> décembre 2023, 15 fév. : 76 j ;
        # trimestrielle -> dernier trimestre observé = octobre 2023 : 137 j
        assert _records(result, 'frequency', 'period_start', 'delay') == _expected(
            ('mensuelle', 'monthly', TS('2023-12-01'), 76.0),
            ('trimestrielle', 'quarterly', TS('2023-10-01'), 137.0),
        )

    @pytest.mark.parametrize(
        ('anchored', 'last_observation'),
        [
            (lambda df: df, '2023-04-01'),
            (perturb.to_period_end, '2023-04-30'),
        ],
        ids=['start-anchored', 'end-anchored'],
    )
    def test_period_position_of_the_index_does_not_change_the_period(self, monthly_series, anchored, last_observation):
        """Month-start and month-end labels designate the same period, hence the same delay."""
        result = compare_and_detect_delays(anchored(monthly_series), None, '2023-06-15')

        assert result[['observation_date', 'period_start', 'period_end', 'delay']].iloc[0].tolist() == [
            TS(last_observation), TS('2023-04-01'), TS('2023-05-01'), 75.0,
        ]

    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_undetectable_frequency_gives_a_row_without_period_or_delay(self, reference_point):
        """A single observation has no detectable frequency: the row is kept, without period nor delay."""
        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(_series([100.0]), None, '2023-02-15', reference_point=reference_point)

        row = result.iloc[0]
        assert list(result.index) == ['PIB'] and row['observation_date'] == TS('2023-01-01')
        assert pd.isna(row['frequency'])
        assert pd.isna(row['period_start']) and pd.isna(row['period_end']) and pd.isna(row['delay'])
        assert (row['unit'], row['reference_point'], row['has_changes']) == ('day', reference_point, True)

    def test_undetectable_frequency_warning_names_the_column(self):
        """The warning points at the variable whose frequency could not be detected."""
        with pytest.warns(UserWarning, match=r"frequency could not be detected for \['PIB'\]"):
            compare_and_detect_delays(_series([100.0]), None, '2023-02-15')

    def test_one_undetectable_entity_does_not_spoil_the_panel(self):
        """An entity with a single observation is returned with a missing frequency; the others are computed."""
        panel = _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'B': {'PIB': [1.0, np.nan, np.nan, np.nan]}})

        with pytest.warns(UserWarning, match=r"\('B', 'PIB'\)") as caught:
            result = compare_and_detect_delays(panel, None, '2023-06-15')

        rows = {key: row for key, row in result.iterrows()}
        assert rows[('A', 'PIB')][['frequency', 'delay']].tolist() == ['monthly', 75.0]
        assert pd.isna(rows[('B', 'PIB')]['frequency']) and pd.isna(rows[('B', 'PIB')]['delay'])
        # Le message ne cite que le couple indétectable
        assert "('A', 'PIB')" not in str(caught[0].message)

    def test_undetectable_frequency_in_comparison(self):
        """The same holds on the comparison path: a lone new observation is reported without delay."""
        data = _series([1.0, np.nan, np.nan, np.nan])
        existing = _series([np.nan] * 4)

        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert list(result.index) == ['PIB'] and pd.isna(result['delay'].iloc[0])

    def test_single_observation_series_gives_a_row_without_delay(self):
        """A frame cut down to its first row behaves like any single observation."""
        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(
                perturb.single_observation(_series([1.0, 2.0, 3.0, 4.0])), None, '2023-06-15',
            )

        assert result['frequency'].isna().all() and pd.isna(result['delay'].iloc[0])


class TestDelayUnitResolution:
    """The delay units rest on ``utils.duration``; only day, second and microsecond are accepted."""

    @pytest.mark.parametrize('unit', ['h', 'hour', 'W', 'ms', 'days', 'D ', '', None, 3])
    def test_other_units_are_rejected_with_the_same_message(self, unit):
        with pytest.raises(ValueError, match="Unit must be one of"):
            compare_and_detect_delays(_series([1.0, 2.0, 3.0]), download_date='2023-06-15', delay_unit=unit)
