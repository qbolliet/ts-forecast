"""Unit tests for ``tsforecast.utils.time.resolution``.

Covers ``as_exact_unit``: exact change of unit between every pair of pandas
units for a ``Timestamp`` and a ``DatetimeIndex`` (same instants, time zone,
name and frequency kept, ``NaT`` and empty index), widening to the coarsest
finer unit when a date has a finer part (never a rounding), the refusal of a
date outside the nanosecond range, invalid units and types, and the export by
``tsforecast.utils.time``.
"""
# Importation des modules
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime

# Résolutions de dates prises en charge par pandas
from tests.support.perturbations import DATETIME_UNITS

# Fonction à tester
import tsforecast.utils.time as time_package
from tsforecast.utils.time.resolution import as_exact_unit


def _nanoseconds(dates) -> list:
    """Instants of a ``Timestamp`` or of a ``DatetimeIndex``, in integer nanoseconds."""
    index = pd.DatetimeIndex([dates]) if isinstance(dates, pd.Timestamp) else dates
    return index.as_unit('ns').asi8.tolist()


class TestExactConversion:
    """Dates representable at the target unit are stored at that very unit."""

    @pytest.mark.parametrize('target', DATETIME_UNITS)
    @pytest.mark.parametrize('source', DATETIME_UNITS)
    def test_timestamp_between_every_pair_of_units(self, source, target):
        """A date to the second: target unit, same instant."""
        date = pd.Timestamp('2024-02-29 13:47:12').as_unit(source)

        result = as_exact_unit(date, target)

        assert isinstance(result, pd.Timestamp)
        assert result.unit == target
        assert _nanoseconds(result) == _nanoseconds(date)

    @pytest.mark.parametrize('target', DATETIME_UNITS)
    @pytest.mark.parametrize('source', DATETIME_UNITS)
    def test_index_between_every_pair_of_units(self, source, target):
        """A monthly grid: target unit, same instants, name and frequency kept."""
        index = pd.date_range('2024-01-01', periods=4, freq='MS', name='date').as_unit(source)

        result = as_exact_unit(index, target)

        assert isinstance(result, pd.DatetimeIndex)
        assert result.unit == target
        assert _nanoseconds(result) == _nanoseconds(index)
        assert result.name == 'date' and result.freq == index.freq

    @pytest.mark.parametrize('target', DATETIME_UNITS)
    def test_time_zone_is_kept(self, target):
        """A time-zone-aware index keeps its zone and its instants."""
        index = pd.date_range('2024-03-30', periods=3, freq='D', tz='Europe/Paris')

        result = as_exact_unit(index, target)

        assert str(result.tz) == 'Europe/Paris'
        assert str(result.dtype) == f'datetime64[{target}, Europe/Paris]'
        assert (result == index).all()

    def test_missing_dates_and_empty_index(self):
        """``NaT`` stays missing; an empty index takes the target unit."""
        with_missing = pd.DatetimeIndex(['2024-01-01', None]).as_unit('ns')

        result = as_exact_unit(with_missing, 's')

        assert result.unit == 's'
        assert result.isna().tolist() == [False, True]
        assert as_exact_unit(pd.DatetimeIndex([]), 'ms').unit == 'ms'


class TestWidening:
    """A date with a part finer than the target unit is never rounded."""

    @pytest.mark.parametrize(
        'date, expected',
        [
            ('2024-01-01 00:00:00.5', 'ms'),
            ('2024-01-01 00:00:00.000001', 'us'),
            ('2024-01-01 00:00:00.000000001', 'ns'),
        ],
        ids=['half-second', 'microsecond', 'nanosecond'],
    )
    def test_coarsest_finer_unit_holding_the_date(self, date, expected):
        """From seconds: the first finer unit holding the date exactly."""
        timestamp = pd.Timestamp(date).as_unit('ns')

        result = as_exact_unit(timestamp, 's')

        assert result.unit == expected
        assert result == timestamp

    def test_one_fine_date_widens_the_whole_index(self):
        """An index has one unit: a single half-second moves every date to milliseconds."""
        index = pd.DatetimeIndex(['2024-01-01', '2024-01-02 00:00:00.5'])

        result = as_exact_unit(index, 's')

        assert result.unit == 'ms'
        assert _nanoseconds(result) == _nanoseconds(index)

    def test_finer_target_than_needed_is_kept(self):
        """The target unit is a floor: a date in seconds asked in microseconds stays in microseconds."""
        assert as_exact_unit(pd.Timestamp('2024-01-01').as_unit('s'), 'us').unit == 'us'


class TestRejections:
    """Out-of-range dates, unknown units and other types are refused."""

    def test_date_outside_the_nanosecond_range_is_refused(self):
        """2262-05-01 has no nanosecond representation (range ends on 2262-04-11)."""
        with pytest.raises(OutOfBoundsDatetime):
            as_exact_unit(pd.Timestamp('2262-05-01'), 'ns')

    @pytest.mark.parametrize('unit', ['D', 'm', 'h', 'NS', None])
    def test_unknown_unit_raises_value_error(self, unit):
        """Only the four pandas storage units are accepted."""
        with pytest.raises(ValueError, match='unit must be one of'):
            as_exact_unit(pd.Timestamp('2024-01-01'), unit)

    @pytest.mark.parametrize(
        'dates',
        [
            pd.Series(pd.to_datetime(['2024-01-01'])),
            pd.period_range('2024-01', periods=2, freq='M'),
            '2024-01-01',
            pd.NaT,
        ],
        ids=['series', 'period-index', 'string', 'nat'],
    )
    def test_other_types_raise_type_error(self, dates):
        """A Series, a PeriodIndex, a string or a scalar ``NaT`` are not dates to cast."""
        with pytest.raises(TypeError, match='must be a pd.Timestamp or a pd.DatetimeIndex'):
            as_exact_unit(dates, 's')


class TestPackageExport:
    """``as_exact_unit`` is part of the public surface of ``tsforecast.utils.time``."""

    def test_exported_by_the_package(self):
        """Listed in ``__all__`` and re-exported as the very function of the module."""
        assert 'as_exact_unit' in time_package.__all__
        assert time_package.as_exact_unit is as_exact_unit
