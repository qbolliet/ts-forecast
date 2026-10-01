"""Unit tests for tsforecast.utils._constants.

The module is the single source of truth for several tables that used to be
duplicated across sub-packages: these tests lock the invariants that keep them
consistent with each other.
"""
import pandas as pd
import pytest

from tsforecast.utils._constants import (
    BLOCK_START_FREQUENCIES,
    CALENDAR_SUBPERIODS,
    CONVERSION_FACTORS_TO_SECONDS,
    INTRADAY_UNITS,
    MONTH_ABBREVIATIONS,
    MONTH_BASED_FREQUENCIES,
    NON_PERIOD_FREQUENCIES,
    POSITION_AWARE_FREQUENCIES,
    SUBDAILY_NS,
    WEEKDAY_ABBREVIATIONS,
)


class TestCalendarVocabulary:
    def test_month_abbreviations_follow_pandas_order(self):
        assert MONTH_ABBREVIATIONS == tuple(m.upper() for m in pd.date_range('2023-01-01', periods=12, freq='MS').strftime('%b'))

    def test_weekday_abbreviations_follow_pandas_order(self):
        assert WEEKDAY_ABBREVIATIONS == tuple(d.upper() for d in pd.date_range('2023-01-02', periods=7).strftime('%a'))


class TestFrequencyClassification:
    def test_position_aware_is_month_based_plus_semi_monthly(self):
        assert POSITION_AWARE_FREQUENCIES == {'M', 'Q', 'Y', 'SM'}

    def test_block_start_and_position_aware_are_disjoint(self):
        assert not BLOCK_START_FREQUENCIES & POSITION_AWARE_FREQUENCIES

    @pytest.mark.parametrize('base', ['W', 'B'])
    def test_week_and_business_day_never_take_a_position_suffix(self, base):
        assert base not in POSITION_AWARE_FREQUENCIES

    def test_week_is_neither_position_aware_nor_block_start(self):
        # Ancre hebdomadaire = jour de la semaine, ni position S/E ni début de bloc
        assert 'W' not in BLOCK_START_FREQUENCIES

    def test_block_start_covers_days_and_subdaily_units(self):
        assert BLOCK_START_FREQUENCIES == {'D', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}

    def test_non_period_frequencies_are_position_aware(self):
        assert NON_PERIOD_FREQUENCIES <= POSITION_AWARE_FREQUENCIES

    def test_month_based_frequencies_have_a_month_count(self):
        assert MONTH_BASED_FREQUENCIES == {'M': 1, 'Q': 3, 'Y': 12}


class TestDurationTables:
    def test_subdaily_ns_matches_pandas_timedelta(self):
        for unit, nanoseconds in SUBDAILY_NS.items():
            assert pd.Timedelta(1, unit=unit).value == nanoseconds

    def test_intraday_units_sorted_from_largest_to_smallest(self):
        sizes = [size for _, size in INTRADAY_UNITS]
        assert sizes == sorted(sizes, reverse=True)
        assert [unit for unit, _ in INTRADAY_UNITS] == ['h', 'min', 's', 'ms', 'us', 'ns']

    def test_seconds_table_is_consistent_with_subdaily_ns(self):
        for unit, nanoseconds in SUBDAILY_NS.items():
            assert CONVERSION_FACTORS_TO_SECONDS[unit] == pytest.approx(nanoseconds / 1e9)

    def test_seconds_table_covers_every_classified_base(self):
        known = set(BLOCK_START_FREQUENCIES) | set(POSITION_AWARE_FREQUENCIES) | {'W'}
        assert known <= set(CONVERSION_FACTORS_TO_SECONDS)

    def test_calendar_subperiods_are_ordered_coarse_to_fine(self):
        for coarse, fine in CALENDAR_SUBPERIODS:
            assert CONVERSION_FACTORS_TO_SECONDS[coarse] > CONVERSION_FACTORS_TO_SECONDS[fine]
