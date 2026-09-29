"""Unit tests for ``tsforecast.utils.time.utils``.

Covers ``resolve_date`` (``'today'``, implicit / explicit formats, ``datetime`` /
``Timestamp`` / ``str`` inputs, invalid values), the ``timeseries_to_string`` /
``string_to_timeseries`` pair (round trip, ``NaT``, time zones) and the period
helpers ``get_period_start``, ``get_period_end`` and ``get_period_boundaries``
(golden table over D, W, SM, M, Q, Y, h, min, s and us, dates already on a
period boundary, leap day, year rollover; anchored (``W-WED``, ``QE-JAN``,
``YS-JUL``) and multiplied (``2MS``, ``3D``, ``2h``) frequencies, the ``origin``
grid; ``ns`` / ``ms`` precision, time zones and DST; ``Period`` / ``datetime64``
inputs; invalid frequencies, anchors and dates; ``start <= date < end``
properties).

Fixes of ANO-UTILS-018 to ANO-UTILS-023 (see ``tests/ANOMALIES.md``) are covered
by ``TestNanosecond``, ``TestMillisecond``, ``TestTimezone``, ``TestPeriodInput``,
``TestPeriodAnchorsAndMultipliers`` and ``TestResolveDate``.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
# Module de gestion des dates
from datetime import date as date_type, datetime, timedelta
# Module de test
import pytest

# Fonctions à tester
from tsforecast.utils.time.utils import (
    get_period_boundaries,
    get_period_end,
    get_period_start,
    resolve_date,
    string_to_timeseries,
    timeseries_to_string,
)

class TestPeriodFunctions:
    """Test suite for period start and end functions."""

    def test_intraday_frequencies(self):
        """Test intraday frequency handling (hourly, minute, second)."""
        test_date = datetime(2023, 6, 15, 14, 35, 47, 123456)
        
        # Test horaire
        assert get_period_start(test_date, 'h') == datetime(2023, 6, 15, 14, 0, 0)
        assert get_period_end(test_date, 'h') == datetime(2023, 6, 15, 15, 0, 0)
        
        # Test minute
        assert get_period_start(test_date, 'min') == datetime(2023, 6, 15, 14, 35, 0)
        assert get_period_end(test_date, 'min') == datetime(2023, 6, 15, 14, 36, 0)
        
        # Test seconde
        assert get_period_start(test_date, 's') == datetime(2023, 6, 15, 14, 35, 47)
        assert get_period_end(test_date, 's') == datetime(2023, 6, 15, 14, 35, 48)
    
    def test_subsecond_frequencies(self):
        """Test sub-second frequency handling (ms, us)."""
        test_date = datetime(2023, 6, 15, 14, 35, 47, 123456)
        
        # Test milliseconde (123.456 ms)
        start_ms = get_period_start(test_date, 'ms')
        assert start_ms.microsecond == 123000  # Arrondi à la milliseconde
        
        # Test microseconde
        start_us = get_period_start(test_date, 'us')
        assert start_us.microsecond == 123456
        
    def test_daily_frequencies(self):
        """Test daily and business daily frequencies."""
        test_date = datetime(2023, 6, 15, 14, 30, 0)
        
        # Test jour calendaire
        assert get_period_start(test_date, 'D') == datetime(2023, 6, 15, 0, 0, 0)
        assert get_period_end(test_date, 'D') == datetime(2023, 6, 16, 0, 0, 0)
        
        # Test jour ouvré (même logique)
        assert get_period_start(test_date, 'B') == datetime(2023, 6, 15, 0, 0, 0)
        assert get_period_end(test_date, 'B') == datetime(2023, 6, 16, 0, 0, 0)
    
    def test_weekly_frequency(self):
        """Test weekly frequency (Monday-based)."""
        # Jeudi 15 juin 2023
        test_date = datetime(2023, 6, 15)
        
        # La semaine commence le lundi (12 juin)
        assert get_period_start(test_date, 'W') == datetime(2023, 6, 12, 0, 0, 0)
        # La semaine se termine le lundi suivant (19 juin)
        assert get_period_end(test_date, 'W') == datetime(2023, 6, 19, 0, 0, 0)
    
    def test_semi_monthly_frequency(self):
        """Test semi-monthly frequency (1-15, 16-end)."""
        # Première quinzaine
        date1 = datetime(2023, 6, 10)
        assert get_period_start(date1, 'SM') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(date1, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
        
        # Deuxième quinzaine
        date2 = datetime(2023, 6, 20)
        assert get_period_start(date2, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
        assert get_period_end(date2, 'SM') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Cas limite : le 15 est dans la première quinzaine
        date3 = datetime(2023, 6, 15)
        assert get_period_start(date3, 'SM') == datetime(2023, 6, 1, 0, 0, 0)
        
        # Cas limite : le 16 est dans la deuxième quinzaine
        date4 = datetime(2023, 6, 16)
        assert get_period_start(date4, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
    
    def test_monthly_frequency(self):
        """Test monthly frequency."""
        test_date = datetime(2023, 6, 15)
        
        assert get_period_start(test_date, 'M') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(test_date, 'M') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Test décembre -> janvier
        dec_date = datetime(2023, 12, 15)
        assert get_period_end(dec_date, 'M') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_quarterly_frequency(self):
        """Test quarterly frequency."""
        # Q1 : janvier-mars
        q1_date = datetime(2023, 2, 15)
        assert get_period_start(q1_date, 'Q') == datetime(2023, 1, 1, 0, 0, 0)
        assert get_period_end(q1_date, 'Q') == datetime(2023, 4, 1, 0, 0, 0)
        
        # Q2 : avril-juin
        q2_date = datetime(2023, 5, 15)
        assert get_period_start(q2_date, 'Q') == datetime(2023, 4, 1, 0, 0, 0)
        assert get_period_end(q2_date, 'Q') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Q3 : juillet-septembre
        q3_date = datetime(2023, 8, 15)
        assert get_period_start(q3_date, 'Q') == datetime(2023, 7, 1, 0, 0, 0)
        assert get_period_end(q3_date, 'Q') == datetime(2023, 10, 1, 0, 0, 0)
        
        # Q4 : octobre-décembre
        q4_date = datetime(2023, 11, 15)
        assert get_period_start(q4_date, 'Q') == datetime(2023, 10, 1, 0, 0, 0)
        assert get_period_end(q4_date, 'Q') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_annual_frequency(self):
        """Test annual frequency."""
        test_date = datetime(2023, 6, 15)
        
        assert get_period_start(test_date, 'Y') == datetime(2023, 1, 1, 0, 0, 0)
        assert get_period_end(test_date, 'Y') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_literal_frequency_names(self):
        """Test using literal frequency names instead of pandas codes."""
        test_date = datetime(2023, 6, 15)
        
        # Test avec noms littéraux
        assert get_period_start(test_date, 'monthly') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_start(test_date, 'quarterly') == datetime(2023, 4, 1, 0, 0, 0)
        assert get_period_start(test_date, 'annual') == datetime(2023, 1, 1, 0, 0, 0)
    
    def test_pandas_timestamp_input(self):
        """Test using pandas Timestamp as input."""
        test_date = pd.Timestamp('2023-06-15 14:30:00')
        
        assert get_period_start(test_date, 'M') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(test_date, 'M') == datetime(2023, 7, 1, 0, 0, 0)
    
    def test_get_period_boundaries(self):
        """Test combined period range function."""
        test_date = datetime(2023, 6, 15)
        
        start, end = get_period_boundaries(test_date, 'M')
        assert start == datetime(2023, 6, 1, 0, 0, 0)
        assert end == datetime(2023, 7, 1, 0, 0, 0)
    
    def test_unsupported_frequency_error(self):
        """Test error handling for unsupported frequencies."""
        test_date = datetime(2023, 6, 15)
        
        with pytest.raises(ValueError, match="Unsupported frequency"):
            get_period_start(test_date, 'invalid_freq')
        
        with pytest.raises(ValueError, match="Unsupported frequency"):
            get_period_end(test_date, 'invalid_freq')



# =============================================================================
# resolve_date
# =============================================================================


class TestResolveDate:
    """``resolve_date`` turns ``'today'``, a string or a datetime into a datetime."""

    @pytest.mark.parametrize("keyword", ["today", "Today", "TODAY"], ids=str)
    def test_today_is_case_insensitive_and_current(self, keyword):
        """The keyword ``'today'`` yields the current instant, whatever its case."""
        # Encadrement de l'instant courant : le résultat doit être compris entre
        # deux lectures de l'horloge
        before = datetime.now()
        result = resolve_date(keyword)
        after = datetime.now()

        assert before <= result <= after

    def test_today_ignores_format(self):
        """A ``format`` is not applied to the keyword ``'today'``."""
        assert isinstance(resolve_date("today", format="%Y"), datetime)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("2023-06-15", datetime(2023, 6, 15)),
            ("2023/06/15", datetime(2023, 6, 15)),
            ("20230615", datetime(2023, 6, 15)),
            ("2023-06-15 14:30:45", datetime(2023, 6, 15, 14, 30, 45)),
            ("2023-06-15T14:30:45", datetime(2023, 6, 15, 14, 30, 45)),
            ("15 June 2023", datetime(2023, 6, 15)),
            # Date ambiguë : pandas interprète le premier champ comme le mois
            ("06/07/2023", datetime(2023, 6, 7)),
        ],
        ids=["iso", "slashes", "compact", "iso-time", "iso-T", "literal-month", "ambiguous-month-first"],
    )
    def test_implicit_format_is_inferred(self, text, expected):
        """A string without ``format`` is parsed with pandas' inference."""
        assert resolve_date(text) == expected

    @pytest.mark.parametrize(
        "text, fmt, expected",
        [
            ("06/15/2023", "%m/%d/%Y", datetime(2023, 6, 15)),
            # Même chaîne que le cas ambigu ci-dessus : le format tranche vers le 6 juillet
            ("06/07/2023", "%d/%m/%Y", datetime(2023, 7, 6)),
            ("20230615", "%Y%m%d", datetime(2023, 6, 15)),
            ("2023-06-15 14:30", "%Y-%m-%d %H:%M", datetime(2023, 6, 15, 14, 30)),
        ],
        ids=["month-first", "day-first", "compact", "with-time"],
    )
    def test_explicit_format_is_applied(self, text, fmt, expected):
        """An explicit ``format`` overrides pandas' inference."""
        assert resolve_date(text, format=fmt) == expected

    def test_string_result_is_plain_datetime(self):
        """A parsed string is returned as a ``datetime``, not a ``Timestamp``."""
        assert type(resolve_date("2023-06-15")) is datetime

    def test_string_with_offset_keeps_it(self):
        """An ISO string carrying a UTC offset yields an aware datetime."""
        result = resolve_date("2023-06-15T10:00:00+02:00")

        assert result.utcoffset() == timedelta(hours=2)

    def test_datetime_is_returned_unchanged(self):
        """A ``datetime`` is returned as is (same object)."""
        value = datetime(2023, 6, 15, 14, 30)

        assert resolve_date(value) is value

    def test_datetime_ignores_format(self):
        """A ``format`` is not applied to a ``datetime`` input."""
        value = datetime(2023, 6, 15, 14, 30)

        assert resolve_date(value, format="%d/%m/%Y") == value

    def test_timestamp_is_returned_unchanged(self):
        """A ``Timestamp`` (a ``datetime`` subclass) is returned as is."""
        value = pd.Timestamp("2023-06-15 14:30")

        assert resolve_date(value) is value

    @pytest.mark.parametrize(
        "text, fmt",
        [
            ("not a date", None),
            ("2023-02-30", None),
            ("2023-06-15", "%d/%m/%Y"),
        ],
        ids=["garbage", "impossible-day", "format-mismatch"],
    )
    def test_invalid_string_raises_value_error(self, text, fmt):
        """An unparseable string, or one not matching ``format``, raises ``ValueError``."""
        with pytest.raises(ValueError):
            resolve_date(text, format=fmt)

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, None, date_type(2023, 6, 15)],
        ids=["int", "float", "none", "date-not-datetime"],
    )
    def test_unsupported_type_raises_value_error(self, value):
        """A value that is neither ``'today'``, a string nor a datetime is rejected."""
        with pytest.raises(ValueError, match="must be 'today'"):
            resolve_date(value)

    @pytest.mark.parametrize("value", ["", pd.NaT], ids=["empty-string", "nat"])
    def test_unresolvable_value_raises_instead_of_returning_nat(self, value):
        """An empty string or ``NaT`` is not a date: ``ValueError``, never ``NaT`` (ANO-UTILS-022)."""
        with pytest.raises(ValueError):
            resolve_date(value)


# =============================================================================
# timeseries_to_string / string_to_timeseries
# =============================================================================


class TestTimeseriesToString:
    """``timeseries_to_string`` formats the datetime index of a series."""

    def test_default_format_is_iso_date(self):
        """The default format is ``%Y-%m-%d``; the time of day is dropped."""
        ts = pd.Series([1, 2], index=pd.to_datetime(["2023-01-01 10:30", "2023-01-02 23:59"]))

        assert timeseries_to_string(ts).index.tolist() == ["2023-01-01", "2023-01-02"]

    def test_custom_format(self):
        """A custom ``format`` drives the index labels."""
        ts = pd.Series([1], index=pd.to_datetime(["2023-06-15 14:30"]))

        assert timeseries_to_string(ts, "%d/%m/%Y %H:%M").index.tolist() == ["15/06/2023 14:30"]

    def test_values_are_preserved(self):
        """Values, ``NaN`` included, are untouched."""
        ts = pd.Series([1.5, np.nan], index=pd.date_range("2023-01-01", periods=2))

        pd.testing.assert_series_equal(
            timeseries_to_string(ts).reset_index(drop=True), ts.reset_index(drop=True)
        )

    def test_name_is_preserved(self):
        """The series name, accents and spaces included, is untouched."""
        ts = pd.Series([1, 2], index=pd.date_range("2023-01-01", periods=2), name="série é")

        assert timeseries_to_string(ts).name == "série é"

    def test_input_is_not_modified(self):
        """The input series keeps its ``DatetimeIndex``."""
        ts = pd.Series([1, 2], index=pd.date_range("2023-01-01", periods=2))

        timeseries_to_string(ts)

        assert isinstance(ts.index, pd.DatetimeIndex)

    def test_empty_series(self):
        """An empty series gives an empty series."""
        ts = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

        assert len(timeseries_to_string(ts)) == 0

    def test_nat_becomes_missing_label(self):
        """A ``NaT`` in the index becomes a missing label (no string)."""
        ts = pd.Series([1, 2], index=pd.DatetimeIndex(["2023-01-01", None]))

        assert pd.isna(timeseries_to_string(ts).index[1])


class TestStringToTimeseries:
    """``string_to_timeseries`` parses the string index of a series."""

    def test_inferred_format(self):
        """Without ``format`` the labels are inferred as ISO dates."""
        ts = pd.Series([1, 2], index=["2023-01-01", "2023-01-02"])

        assert string_to_timeseries(ts).index.tolist() == [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
        ]

    def test_explicit_format(self):
        """An explicit ``format`` resolves day-first ambiguity."""
        ts = pd.Series([1], index=["01/02/2023"])

        # Valeur d'or : jour d'abord, donc le 1er février
        assert string_to_timeseries(ts, format="%d/%m/%Y").index[0] == pd.Timestamp("2023-02-01")

    def test_result_index_is_datetime(self):
        """The resulting index is a ``DatetimeIndex``."""
        ts = pd.Series([1], index=["2023-01-01"])

        assert isinstance(string_to_timeseries(ts).index, pd.DatetimeIndex)

    def test_missing_label_becomes_nat(self):
        """A missing label (``None``) becomes ``NaT``."""
        ts = pd.Series([1, 2], index=["2023-01-01", None])

        assert pd.isna(string_to_timeseries(ts).index[1])

    @pytest.mark.parametrize(
        "label, fmt",
        [("xx", None), ("2023-01-01", "%d/%m/%Y")],
        ids=["garbage", "format-mismatch"],
    )
    def test_invalid_label_raises_value_error(self, label, fmt):
        """An unparseable label, or one not matching ``format``, raises ``ValueError``."""
        with pytest.raises(ValueError):
            string_to_timeseries(pd.Series([1], index=[label]), format=fmt)

    def test_values_are_preserved(self):
        """Values are untouched."""
        ts = pd.Series([3.0, 4.0], index=["2023-01-01", "2023-01-02"])

        assert string_to_timeseries(ts).values.tolist() == [3.0, 4.0]

    def test_name_is_preserved(self):
        """The series name is untouched."""
        ts = pd.Series([1], index=["2023-01-01"], name="ma série")

        assert string_to_timeseries(ts).name == "ma série"

    def test_empty_series(self):
        """An empty series gives an empty series with a ``DatetimeIndex``."""
        ts = pd.Series([], index=pd.Index([], dtype=object), dtype=float)

        assert isinstance(string_to_timeseries(ts).index, pd.DatetimeIndex)


class TestStringRoundTrip:
    """``string_to_timeseries(timeseries_to_string(ts, f), f)`` restores ``ts``."""

    @staticmethod
    def _assert_roundtrip(ts: pd.Series, fmt: str) -> None:
        """Assert that a format/parse round trip restores the series."""
        restored = string_to_timeseries(timeseries_to_string(ts, fmt), fmt)
        # La fréquence de l'index n'est pas conservée par le passage en chaîne
        pd.testing.assert_series_equal(restored, ts, check_freq=False)

    @pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d", "%m-%d-%Y"], ids=str)
    def test_daily_index(self, fmt):
        """Daily index: identity for any date-only format."""
        ts = pd.Series(np.arange(5.0), index=pd.date_range("2023-02-27", periods=5), name="x")

        self._assert_roundtrip(ts, fmt)

    def test_intraday_index_with_time_format(self):
        """Hourly index: identity as long as the format carries the hour."""
        ts = pd.Series(np.arange(4.0), index=pd.date_range("2023-06-15 22:00", periods=4, freq="h"))

        self._assert_roundtrip(ts, "%Y-%m-%d %H:%M:%S")

    def test_unsorted_index_with_duplicates(self):
        """Neither order nor duplicates are altered (no sort, no dedup)."""
        index = pd.to_datetime(["2023-03-01", "2023-01-01", "2023-03-01", "2023-02-01"])
        ts = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

        self._assert_roundtrip(ts, "%Y-%m-%d")

    def test_leap_day(self):
        """29 February survives the round trip."""
        ts = pd.Series([1.0], index=pd.to_datetime(["2024-02-29"]))

        self._assert_roundtrip(ts, "%Y-%m-%d")

    def test_default_format_drops_time_of_day(self):
        """With the default date-only format the round trip truncates to midnight."""
        ts = pd.Series([1.0], index=pd.to_datetime(["2023-06-15 14:30"]))

        restored = string_to_timeseries(timeseries_to_string(ts))

        assert restored.index[0] == pd.Timestamp("2023-06-15")

    def test_nat_survives_round_trip(self):
        """A ``NaT`` label comes back as ``NaT`` at the same position."""
        ts = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2023-01-01", None, "2023-01-03"]))

        restored = string_to_timeseries(timeseries_to_string(ts))

        assert restored.index.isna().tolist() == [False, True, False]

    def test_timezone_with_offset_format_keeps_instants(self):
        """With ``%z`` in the format, the absolute instants are restored."""
        index = pd.date_range("2023-01-01", periods=3, freq="D", tz="Europe/Paris")
        ts = pd.Series([1.0, 2.0, 3.0], index=index)
        fmt = "%Y-%m-%d %H:%M:%S%z"

        restored = string_to_timeseries(timeseries_to_string(ts, fmt), fmt)

        # Comparaison des instants absolus : le fuseau nommé est réduit à un décalage fixe
        assert (restored.index == index).all()

    def test_timezone_without_offset_format_gives_naive_wall_time(self):
        """Without ``%z`` the time zone is lost: naive local wall-clock time."""
        index = pd.date_range("2023-06-15 09:00", periods=2, freq="h", tz="Europe/Paris")
        ts = pd.Series([1.0, 2.0], index=index)
        fmt = "%Y-%m-%d %H:%M:%S"

        restored = string_to_timeseries(timeseries_to_string(ts, fmt), fmt)

        assert restored.index.tolist() == [pd.Timestamp("2023-06-15 09:00"), pd.Timestamp("2023-06-15 10:00")]



# =============================================================================
# get_period_start / get_period_end / get_period_boundaries
# =============================================================================

# Écritures équivalentes d'une même unité : code pandas, positions début / fin, nom
# littéral. Les positions (MS / ME, QS / QE, YS / YE) n'ont pas d'effet sur les bornes ;
# les ancres (W-WED, QE-JAN...) et les multiplicateurs ont leurs propres tables plus bas.
_MONTHLY = ["M", "MS", "ME", "monthly"]
_QUARTERLY = ["Q", "QS", "QE", "quarterly"]
_ANNUAL = ["Y", "YS", "YE", "annual"]
_WEEKLY = ["W", "W-SUN", "weekly"]


def _table(freqs, rows):
    """Expand ``(label, date, start, end)`` rows over equivalent frequency spellings.

    Args:
        freqs: Equivalent frequency spellings sharing the same boundaries.
        rows: Tuples ``(label, date, expected_start, expected_end)``.

    Returns:
        List of ``pytest.param(date, freq, start, end)`` with readable ids.
    """
    return [
        pytest.param(d, f, s, e, id=f"{f}-{label}")
        for f in freqs
        for label, d, s, e in rows
    ]


# Valeurs d'or calculées à la main : chaque ligne (date, début, fin) suit la
# convention [début, fin) où fin est la première date hors de la période.
_GOLDEN = [
    # --- Jour : minuit à minuit, dont la date déjà en bord de période
    *_table(["D", "daily", "B", "business_daily"], [
        ("midday", datetime(2023, 6, 15, 14, 30), datetime(2023, 6, 15), datetime(2023, 6, 16)),
        ("on-start-boundary", datetime(2023, 6, 15), datetime(2023, 6, 15), datetime(2023, 6, 16)),
        ("last-second", datetime(2023, 6, 15, 23, 59, 59), datetime(2023, 6, 15), datetime(2023, 6, 16)),
        ("year-end", datetime(2023, 12, 31, 12), datetime(2023, 12, 31), datetime(2024, 1, 1)),
        ("leap-day", datetime(2024, 2, 29, 8), datetime(2024, 2, 29), datetime(2024, 3, 1)),
        ("day-before-leap-day", datetime(2024, 2, 28, 8), datetime(2024, 2, 28), datetime(2024, 2, 29)),
        ("non-leap-feb-end", datetime(2023, 2, 28, 8), datetime(2023, 2, 28), datetime(2023, 3, 1)),
    ]),
    # --- Semaine `W` (= `W-SUN`) : du lundi au lundi suivant
    *_table(_WEEKLY, [
        ("thursday", datetime(2023, 6, 15, 9), datetime(2023, 6, 12), datetime(2023, 6, 19)),
        ("monday-on-boundary", datetime(2023, 6, 12), datetime(2023, 6, 12), datetime(2023, 6, 19)),
        ("sunday-last-hours", datetime(2023, 6, 18, 23), datetime(2023, 6, 12), datetime(2023, 6, 19)),
        ("year-end-sunday", datetime(2023, 12, 31), datetime(2023, 12, 25), datetime(2024, 1, 1)),
        ("year-start-monday", datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 8)),
        ("leap-day-thursday", datetime(2024, 2, 29), datetime(2024, 2, 26), datetime(2024, 3, 4)),
    ]),
    # --- Semi-mensuel : 1-15 puis 16-fin de mois
    *_table(["SM", "semi_monthly"], [
        ("first-half", datetime(2023, 6, 10), datetime(2023, 6, 1), datetime(2023, 6, 16)),
        ("15th-still-first-half", datetime(2023, 6, 15, 23), datetime(2023, 6, 1), datetime(2023, 6, 16)),
        ("16th-on-boundary", datetime(2023, 6, 16), datetime(2023, 6, 16), datetime(2023, 7, 1)),
        ("december-second-half", datetime(2023, 12, 20), datetime(2023, 12, 16), datetime(2024, 1, 1)),
        ("leap-day-second-half", datetime(2024, 2, 29), datetime(2024, 2, 16), datetime(2024, 3, 1)),
    ]),
    # --- Mois : 1er du mois au 1er du mois suivant
    *_table(_MONTHLY, [
        ("mid-month", datetime(2023, 6, 15, 14, 30), datetime(2023, 6, 1), datetime(2023, 7, 1)),
        ("on-start-boundary", datetime(2023, 6, 1), datetime(2023, 6, 1), datetime(2023, 7, 1)),
        ("last-second", datetime(2023, 6, 30, 23, 59, 59), datetime(2023, 6, 1), datetime(2023, 7, 1)),
        ("leap-february", datetime(2024, 2, 29), datetime(2024, 2, 1), datetime(2024, 3, 1)),
        ("non-leap-february", datetime(2023, 2, 28), datetime(2023, 2, 1), datetime(2023, 3, 1)),
        ("december-rollover", datetime(2023, 12, 31), datetime(2023, 12, 1), datetime(2024, 1, 1)),
    ]),
    # --- Trimestre : Q1 janv.-mars, Q2 avr.-juin, Q3 juil.-sept., Q4 oct.-déc.
    *_table(_QUARTERLY, [
        ("q1-first-day", datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 4, 1)),
        ("q1-last-day", datetime(2023, 3, 31), datetime(2023, 1, 1), datetime(2023, 4, 1)),
        ("q2-first-day", datetime(2023, 4, 1), datetime(2023, 4, 1), datetime(2023, 7, 1)),
        ("q3-mid", datetime(2023, 8, 15), datetime(2023, 7, 1), datetime(2023, 10, 1)),
        ("q4-last-day-rollover", datetime(2023, 12, 31), datetime(2023, 10, 1), datetime(2024, 1, 1)),
        ("leap-day-in-q1", datetime(2024, 2, 29), datetime(2024, 1, 1), datetime(2024, 4, 1)),
    ]),
    # --- Année : 1er janvier au 1er janvier suivant
    *_table(_ANNUAL, [
        ("mid-year", datetime(2023, 6, 15), datetime(2023, 1, 1), datetime(2024, 1, 1)),
        ("on-start-boundary", datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)),
        ("last-second", datetime(2023, 12, 31, 23, 59, 59), datetime(2023, 1, 1), datetime(2024, 1, 1)),
        ("leap-day", datetime(2024, 2, 29), datetime(2024, 1, 1), datetime(2025, 1, 1)),
    ]),
    # --- Heure : bord inférieur inclus, débordement sur le jour / le mois / l'année suivant
    *_table(["h", "hourly"], [
        ("mid-hour", datetime(2023, 6, 15, 14, 35, 47, 123456), datetime(2023, 6, 15, 14), datetime(2023, 6, 15, 15)),
        ("on-start-boundary", datetime(2023, 6, 15, 14), datetime(2023, 6, 15, 14), datetime(2023, 6, 15, 15)),
        ("day-rollover", datetime(2023, 6, 15, 23, 59), datetime(2023, 6, 15, 23), datetime(2023, 6, 16)),
        ("leap-day-rollover", datetime(2024, 2, 28, 23, 30), datetime(2024, 2, 28, 23), datetime(2024, 2, 29)),
        ("leap-day-to-march", datetime(2024, 2, 29, 23, 30), datetime(2024, 2, 29, 23), datetime(2024, 3, 1)),
        ("year-rollover", datetime(2023, 12, 31, 23, 30), datetime(2023, 12, 31, 23), datetime(2024, 1, 1)),
    ]),
    # --- Minute et seconde
    *_table(["min", "minute"], [
        ("mid-minute", datetime(2023, 6, 15, 14, 35, 47), datetime(2023, 6, 15, 14, 35), datetime(2023, 6, 15, 14, 36)),
        ("hour-rollover", datetime(2023, 6, 15, 14, 59, 30), datetime(2023, 6, 15, 14, 59), datetime(2023, 6, 15, 15)),
        ("year-rollover", datetime(2023, 12, 31, 23, 59, 30), datetime(2023, 12, 31, 23, 59), datetime(2024, 1, 1)),
    ]),
    *_table(["s", "second"], [
        ("mid-second", datetime(2023, 6, 15, 14, 35, 47, 123456), datetime(2023, 6, 15, 14, 35, 47), datetime(2023, 6, 15, 14, 35, 48)),
        ("minute-rollover", datetime(2023, 6, 15, 14, 35, 59, 500000), datetime(2023, 6, 15, 14, 35, 59), datetime(2023, 6, 15, 14, 36)),
        ("year-rollover", datetime(2023, 12, 31, 23, 59, 59, 999999), datetime(2023, 12, 31, 23, 59, 59), datetime(2024, 1, 1)),
    ]),
    # --- Microseconde : période d'une microseconde
    *_table(["us", "microsecond"], [
        ("mid", datetime(2023, 6, 15, 14, 35, 47, 123456), datetime(2023, 6, 15, 14, 35, 47, 123456), datetime(2023, 6, 15, 14, 35, 47, 123457)),
        ("second-rollover", datetime(2023, 6, 15, 14, 35, 47, 999999), datetime(2023, 6, 15, 14, 35, 47, 999999), datetime(2023, 6, 15, 14, 35, 48)),
    ]),
]


class TestPeriodGoldenTable:
    """Hand-computed boundaries: ``[start, end)`` with ``end`` the first date outside."""

    @pytest.mark.parametrize("date, freq, start, end", _GOLDEN)
    def test_period_start(self, date, freq, start, end):
        """``get_period_start`` returns the golden start of the period."""
        assert get_period_start(date, freq) == start

    @pytest.mark.parametrize("date, freq, start, end", _GOLDEN)
    def test_period_end(self, date, freq, start, end):
        """``get_period_end`` returns the golden (exclusive) end of the period."""
        assert get_period_end(date, freq) == end

    @pytest.mark.parametrize("date, freq, start, end", _GOLDEN)
    def test_period_boundaries_pair_start_and_end(self, date, freq, start, end):
        """``get_period_boundaries`` returns ``(start, end)`` in that order."""
        assert get_period_boundaries(date, freq) == (start, end)

    @pytest.mark.parametrize("date, freq, start, end", _GOLDEN)
    def test_timestamp_input_gives_same_boundaries(self, date, freq, start, end):
        """A ``pandas.Timestamp`` input gives the same boundaries as a ``datetime``."""
        assert get_period_boundaries(pd.Timestamp(date), freq) == (start, end)

    @pytest.mark.parametrize("date, freq, start, end", _GOLDEN)
    def test_results_are_timestamps(self, date, freq, start, end):
        """Both boundaries are ``pd.Timestamp`` (a ``datetime`` subclass)."""
        result = get_period_boundaries(date, freq)

        assert [type(b) for b in result] == [pd.Timestamp, pd.Timestamp]


def _ts(text: str, tz: str | None = None) -> pd.Timestamp:
    """Build a ``Timestamp`` from text, optionally localized."""
    return pd.Timestamp(text, tz=tz)


# Jeudi 15 juin 2023 : date de référence des tables d'ancres et de multiplicateurs
_THURSDAY = datetime(2023, 6, 15)

# Ancres : (fréquence, date, début, fin). Valeurs d'or calculées à la main.
_ANCHORED = [
    # Semaine `W-X` : elle se termine le jour X, donc commence le lendemain
    pytest.param("W-SUN", _THURSDAY, "2023-06-12", "2023-06-19", id="W-SUN-monday-start"),
    pytest.param("W-MON", _THURSDAY, "2023-06-13", "2023-06-20", id="W-MON-tuesday-start"),
    pytest.param("W-WED", _THURSDAY, "2023-06-15", "2023-06-22", id="W-WED-thursday-on-boundary"),
    pytest.param("W-THU", _THURSDAY, "2023-06-09", "2023-06-16", id="W-THU-thursday-is-last-day"),
    pytest.param("W-FRI", _THURSDAY, "2023-06-10", "2023-06-17", id="W-FRI-saturday-start"),
    pytest.param("W-SAT", _THURSDAY, "2023-06-11", "2023-06-18", id="W-SAT-sunday-start"),
    # Trimestre : QE-X / Q-X se terminent en X, QS-X commencent en X
    pytest.param("Q-DEC", _THURSDAY, "2023-04-01", "2023-07-01", id="Q-DEC-calendar"),
    pytest.param("QS-JAN", _THURSDAY, "2023-04-01", "2023-07-01", id="QS-JAN-calendar"),
    pytest.param("QS-APR", _THURSDAY, "2023-04-01", "2023-07-01", id="QS-APR-same-quarters-as-JAN"),
    pytest.param("Q-JAN", _THURSDAY, "2023-05-01", "2023-08-01", id="Q-JAN-may-july"),
    pytest.param("QE-JAN", _THURSDAY, "2023-05-01", "2023-08-01", id="QE-JAN-may-july"),
    pytest.param("QS-FEB", _THURSDAY, "2023-05-01", "2023-08-01", id="QS-FEB-equals-QE-JAN"),
    pytest.param("QE-NOV", _THURSDAY, "2023-06-01", "2023-09-01", id="QE-NOV-june-august"),
    pytest.param("QE-JAN", datetime(2023, 1, 15), "2022-11-01", "2023-02-01", id="QE-JAN-year-rollover-back"),
    pytest.param("QE-NOV", datetime(2023, 12, 15), "2023-12-01", "2024-03-01", id="QE-NOV-year-rollover-forward"),
    # Année : YE-X se termine en X, YS-X commence en X
    pytest.param("YS-JUL", _THURSDAY, "2022-07-01", "2023-07-01", id="YS-JUL-fiscal"),
    pytest.param("YE-JUN", _THURSDAY, "2022-07-01", "2023-07-01", id="YE-JUN-equals-YS-JUL"),
    pytest.param("Y-JUN", _THURSDAY, "2022-07-01", "2023-07-01", id="Y-JUN-no-position-means-end"),
    pytest.param("YE-MAR", _THURSDAY, "2023-04-01", "2024-04-01", id="YE-MAR-april-march"),
    pytest.param("YS-JUL", datetime(2023, 7, 1), "2023-07-01", "2024-07-01", id="YS-JUL-on-boundary"),
    pytest.param("YS-JAN", _THURSDAY, "2023-01-01", "2024-01-01", id="YS-JAN-calendar"),
]

# Multiplicateurs, grille alignée sur l'époque Unix (jeudi 1970-01-01, indice de jour 0).
# 2023-06-15 est le jour 19 523 depuis l'époque : 19 523 // 3 = 6 507 -> jour 19 521 = 13 juin ;
# 19 523 // 2 = 9 761 -> jour 19 522 = 14 juin ; la grille bihebdomadaire part du lundi
# 1969-12-29 (jour -3) : (19 523 + 3) // 14 = 1 394 -> lundi 5 juin.
_MULTIPLIED = [
    pytest.param("2MS", _THURSDAY, "2023-05-01", "2023-07-01", id="2MS-may-june"),
    pytest.param("2ME", _THURSDAY, "2023-05-01", "2023-07-01", id="2ME-same-as-2MS"),
    pytest.param("3M", _THURSDAY, "2023-04-01", "2023-07-01", id="3M-april-june"),
    pytest.param("6M", _THURSDAY, "2023-01-01", "2023-07-01", id="6M-first-half"),
    pytest.param("2QS", _THURSDAY, "2023-01-01", "2023-07-01", id="2QS-first-half"),
    pytest.param("2QS-FEB", _THURSDAY, "2023-02-01", "2023-08-01", id="2QS-FEB-anchor-shifts-grid"),
    pytest.param("2YS", _THURSDAY, "2022-01-01", "2024-01-01", id="2YS-even-years"),
    pytest.param("2D", _THURSDAY, "2023-06-14", "2023-06-16", id="2D"),
    pytest.param("3D", _THURSDAY, "2023-06-13", "2023-06-16", id="3D"),
    pytest.param("2W", _THURSDAY, "2023-06-05", "2023-06-19", id="2W-monday-grid"),
    pytest.param("2W-WED", _THURSDAY, "2023-06-08", "2023-06-22", id="2W-WED-thursday-grid"),
    pytest.param("2SM", datetime(2023, 6, 20), "2023-06-01", "2023-07-01", id="2SM-whole-month"),
    pytest.param("2h", datetime(2023, 6, 15, 15, 35), "2023-06-15 14:00", "2023-06-15 16:00", id="2h-even-hours"),
    pytest.param("15min", datetime(2023, 6, 15, 15, 35), "2023-06-15 15:30", "2023-06-15 15:45", id="15min"),
    pytest.param("30s", datetime(2023, 6, 15, 15, 35, 47), "2023-06-15 15:35:30", "2023-06-15 15:36:00", id="30s"),
]


class TestPeriodAnchorsAndMultipliers:
    """Anchors and multipliers of pandas frequencies are honoured."""

    @pytest.mark.parametrize("freq, date, start, end", _ANCHORED)
    def test_anchored_periods(self, freq, date, start, end):
        """Anchored frequencies delimit the periods of their own fiscal calendar."""
        assert get_period_boundaries(date, freq) == (_ts(start), _ts(end))

    @pytest.mark.parametrize("freq, date, start, end", _MULTIPLIED)
    def test_multiplied_periods_aligned_on_epoch(self, freq, date, start, end):
        """``nX`` gives periods of ``n`` units on the epoch-aligned grid."""
        assert get_period_boundaries(date, freq) == (_ts(start), _ts(end))

    @pytest.mark.parametrize("freq", ["2MS", "2QE-JAN", "3D", "2W-WED", "2h", "5min"], ids=str)
    def test_multiplied_period_length_is_n_units(self, freq):
        """A multiplied period spans exactly ``n`` times its base period."""
        value = datetime(2023, 6, 15, 14, 35)
        n = int("".join(c for c in freq if c.isdigit())[:1])
        base = freq[len(str(n)):]

        start, end = get_period_boundaries(value, freq)
        # Chaîne de n périodes de base contiguës depuis le début : leur fin est la fin de la période
        cursor = start
        for _ in range(n):
            cursor = get_period_end(cursor, base)

        assert cursor == end

    @pytest.mark.parametrize("position", ["MS", "ME"], ids=str)
    def test_position_does_not_change_month_periods(self, position):
        """``MS`` and ``ME`` delimit the same months."""
        assert get_period_boundaries(_THURSDAY, position) == (_ts("2023-06-01"), _ts("2023-07-01"))


class TestPeriodOrigin:
    """The optional ``origin`` sets the grid of multiplied frequencies."""

    @pytest.mark.parametrize(
        "freq, date, origin, start, end",
        [
            ("2D", datetime(2023, 6, 4), datetime(2023, 6, 1), "2023-06-03", "2023-06-05"),
            ("2D", datetime(2023, 6, 4), datetime(2023, 6, 2), "2023-06-04", "2023-06-06"),
            ("W", _THURSDAY, datetime(2023, 6, 14), "2023-06-14", "2023-06-21"),
            ("2MS", _THURSDAY, datetime(2023, 2, 10), "2023-06-01", "2023-08-01"),
            ("3M", _THURSDAY, datetime(2023, 2, 1), "2023-05-01", "2023-08-01"),
            ("SM", datetime(2023, 6, 20), datetime(2023, 1, 16), "2023-06-16", "2023-07-01"),
            ("h", datetime(2023, 6, 15, 15, 10), datetime(2023, 6, 15, 14, 30), "2023-06-15 14:30", "2023-06-15 15:30"),
            ("2h", datetime(2023, 6, 15, 17, 10), datetime(2023, 6, 15, 14, 30), "2023-06-15 16:30", "2023-06-15 18:30"),
        ],
        ids=["2D-odd-origin", "2D-even-origin", "W-wednesday-origin", "2MS-february-origin",
             "3M-february-origin", "SM-second-half-origin", "h-half-hour-origin", "2h-half-hour-origin"],
    )
    def test_origin_defines_grid(self, freq, date, origin, start, end):
        """Periods are ``origin + k * n * unit``."""
        assert get_period_boundaries(date, freq, origin=origin) == (_ts(start), _ts(end))

    def test_origin_before_and_after_date_give_same_grid(self):
        """The grid extends on both sides of the origin."""
        early = get_period_boundaries(datetime(2023, 6, 4), "3D", origin=datetime(2023, 6, 20))
        late = get_period_boundaries(datetime(2023, 6, 4), "3D", origin=datetime(2023, 4, 30))

        # Origines distantes de 51 jours (17 x 3) : même grille. Valeur d'or : 20 juin - 18 j = 2 juin
        assert early == late == (_ts("2023-06-02"), _ts("2023-06-05"))

    def test_origin_ignores_anchor_of_frequency(self):
        """With an explicit origin the grid is fixed by it, not by the anchor."""
        assert get_period_start(_THURSDAY, "Q-JAN", origin=datetime(2023, 1, 1)) == _ts("2023-04-01")

    def test_naive_origin_is_read_in_the_time_zone_of_the_date(self):
        """A naive origin is wall-clock time in the zone of an aware date."""
        aware = _ts("2023-06-04 12:00", "Europe/Paris")

        start, _ = get_period_boundaries(aware, "2D", origin=datetime(2023, 6, 1))

        assert start == _ts("2023-06-03", "Europe/Paris")

    def test_aware_origin_is_converted_to_the_time_zone_of_the_date(self):
        """An aware origin is converted before use."""
        aware = _ts("2023-06-04 12:00", "Europe/Paris")
        origin = _ts("2023-05-31 22:00", "UTC")  # 1er juin 00:00 à Paris (UTC+2)

        start, _ = get_period_boundaries(aware, "2D", origin=origin)

        assert start == _ts("2023-06-03", "Europe/Paris")

    def test_nat_origin_raises(self):
        """A ``NaT`` origin is rejected."""
        with pytest.raises(ValueError, match="origin"):
            get_period_start(_THURSDAY, "2D", origin=pd.NaT)


class TestPeriodProperties:
    """Invariants of ``[start, end)`` over a sample of dates."""

    # Échantillon déterministe : dates charnières écrites à la main, complétées par
    # des instants tirés avec une graine fixe (jamais hash())
    _RNG = np.random.default_rng(20240229)
    _SAMPLE = [
        datetime(2023, 1, 1),
        datetime(2023, 12, 31, 23, 59, 59, 999999),
        datetime(2024, 2, 29, 12),
        datetime(2024, 2, 29, 23, 59, 59),
        datetime(2023, 2, 28, 23, 59),
        datetime(2024, 3, 1),
        datetime(2023, 6, 15, 14, 35, 47, 123456),
        datetime(2023, 6, 15),
        datetime(2023, 6, 16),
        datetime(2023, 9, 30, 23, 59, 59),
        datetime(2023, 10, 1),
        datetime(1969, 12, 31, 23, 59),
        datetime(2100, 2, 28, 23, 59),
        *[
            datetime(2019, 1, 1) + timedelta(seconds=int(s))
            for s in _RNG.integers(0, 7 * 365 * 86400, size=7)
        ],
    ]
    _FREQS = [
        "ns", "us", "ms", "s", "min", "h", "D", "B", "W", "SM", "M", "Q", "Y",
        # Ancres et multiplicateurs
        "W-WED", "QE-JAN", "QS-FEB", "YS-JUL", "YE-MAR", "2MS", "3M", "2QS", "2Y", "2SM",
        "3D", "2W", "2h", "15min", "7s",
    ]

    @pytest.mark.parametrize("freq", _FREQS, ids=str)
    @pytest.mark.parametrize("date", _SAMPLE, ids=lambda d: d.isoformat())
    def test_date_lies_in_its_period(self, date, freq):
        """``start <= date < end`` for every date."""
        start, end = get_period_boundaries(date, freq)

        assert start <= date < end

    @pytest.mark.parametrize("freq", _FREQS, ids=str)
    @pytest.mark.parametrize("date", _SAMPLE, ids=lambda d: d.isoformat())
    def test_start_is_idempotent(self, date, freq):
        """The start of a period is its own start."""
        start = get_period_start(date, freq)

        assert get_period_start(start, freq) == start

    @pytest.mark.parametrize("freq", _FREQS, ids=str)
    @pytest.mark.parametrize("date", _SAMPLE, ids=lambda d: d.isoformat())
    def test_periods_are_contiguous(self, date, freq):
        """The exclusive end of a period is the start of the next one."""
        end = get_period_end(date, freq)

        assert get_period_start(end, freq) == end


class TestNanosecond:
    """``'ns'`` periods (ANO-UTILS-018): one nanosecond, in a ``Timestamp``."""

    def test_datetime_period_lasts_one_nanosecond(self):
        """A ``datetime`` (microsecond precision) yields a one-nanosecond period."""
        value = datetime(2023, 6, 15, 1, 2, 3, 456789)

        start, end = get_period_boundaries(value, "ns")

        assert (start, end - start) == (pd.Timestamp(value), pd.Timedelta(1, unit="ns"))

    def test_timestamp_nanoseconds_are_kept(self):
        """The nanoseconds of a ``Timestamp`` are not truncated."""
        value = pd.Timestamp("2023-06-15 01:02:03.456789123")

        assert get_period_boundaries(value, "ns") == (value, pd.Timestamp("2023-06-15 01:02:03.456789124"))

    def test_start_is_input_date(self):
        """The period containing an instant at ns resolution starts at that instant."""
        value = datetime(2023, 6, 15, 1, 2, 3, 456789)

        assert get_period_start(value, "ns") == value


class TestMillisecond:
    """``'ms'`` periods (ANO-UTILS-019): floored, contiguous, one millisecond long."""

    def test_end_is_floored(self):
        """The end is the start plus one millisecond."""
        value = datetime(2023, 6, 15, 14, 35, 47, 123456)

        # Valeur d'or : période [47.123 s ; 47.124 s)
        assert get_period_boundaries(value, "ms") == (
            _ts("2023-06-15 14:35:47.123"),
            _ts("2023-06-15 14:35:47.124"),
        )

    def test_periods_are_contiguous(self):
        """The end of a period is the start of the next one."""
        end = get_period_end(datetime(2023, 6, 15, 14, 35, 47, 123456), "ms")

        assert get_period_start(end, "ms") == end


class TestTimezone:
    """Time zones (ANO-UTILS-023): preserved, wall-clock calendar, DST-aware."""

    _FREQS = [
        "ns", "us", "ms", "s", "min", "h", "D", "B", "W", "SM", "M", "Q", "Y",
        "W-WED", "QE-JAN", "YS-JUL", "2MS", "3D", "2h",
    ]

    @pytest.mark.parametrize("freq", _FREQS, ids=str)
    def test_timezone_is_preserved(self, freq):
        """Both boundaries of an aware date carry its time zone."""
        aware = _ts("2023-06-15 14:35:47.123456", "Europe/Paris")

        start, end = get_period_boundaries(aware, freq)

        # Comparaison des noms de zone : les tzinfo pytz diffèrent selon le décalage (CET / CEST)
        assert (str(start.tz), str(end.tz)) == ("Europe/Paris", "Europe/Paris")

    @pytest.mark.parametrize("freq", _FREQS, ids=str)
    def test_aware_date_lies_in_its_period(self, freq):
        """``start <= date < end`` holds for aware dates (comparable bounds)."""
        aware = _ts("2023-06-15 14:35:47.123456", "Europe/Paris")

        start, end = get_period_boundaries(aware, freq)

        assert start <= aware < end

    def test_naive_input_stays_naive(self):
        """A naive date gives naive boundaries."""
        start, end = get_period_boundaries(datetime(2023, 6, 15, 14, 35), "h")

        assert (start.tzinfo, end.tzinfo) == (None, None)

    def test_calendar_boundaries_follow_the_local_wall_clock(self):
        """The month of a Paris date starts at local midnight (UTC+1 in winter, +2 in summer)."""
        start, end = get_period_boundaries(_ts("2023-03-15 12:00", "Europe/Paris"), "M")

        # Valeur d'or : le 1er mars est à UTC+1, le 1er avril à UTC+2 (passage à l'heure d'été le 26 mars)
        assert (start, end) == (_ts("2023-02-28 23:00", "UTC"), _ts("2023-03-31 22:00", "UTC"))

    def test_day_of_spring_forward_lasts_23_hours(self):
        """26 March 2023 in Paris: local midnight to local midnight is 23 hours."""
        start, end = get_period_boundaries(_ts("2023-03-26 12:00", "Europe/Paris"), "D")

        assert end - start == pd.Timedelta(hours=23)

    def test_day_of_fall_back_lasts_25_hours(self):
        """29 October 2023 in Paris: local midnight to local midnight is 25 hours."""
        start, end = get_period_boundaries(_ts("2023-10-29 12:00", "Europe/Paris"), "D")

        assert end - start == pd.Timedelta(hours=25)

    def test_ambiguous_hour_keeps_its_utc_offset(self):
        """Second occurrence of 02:30 on the fall-back night: the hour stays in UTC+1."""
        # 01:30 UTC = 02:30 à Paris, seconde occurrence (UTC+1, après le retour à l'heure d'hiver)
        second_pass = _ts("2023-10-29 01:30", "UTC").tz_convert("Europe/Paris")

        assert get_period_boundaries(second_pass, "h") == (_ts("2023-10-29 01:00", "UTC"), _ts("2023-10-29 02:00", "UTC"))

    def test_half_hour_offset_zone_hour_starts_on_local_hour(self):
        """In Kolkata (UTC+5:30) an hour starts at ``HH:00`` local time."""
        start, end = get_period_boundaries(_ts("2023-06-15 14:35", "Asia/Kolkata"), "h")

        assert (start, end) == (_ts("2023-06-15 14:00", "Asia/Kolkata"), _ts("2023-06-15 15:00", "Asia/Kolkata"))


class TestPeriodInput:
    """Inputs other than ``datetime``: ``Period`` (ANO-UTILS-020), ``datetime64``, invalid types."""

    @pytest.mark.parametrize(
        "freq, start, end",
        [
            ("us", "2023-06-15 00:00:00", "2023-06-15 00:00:00.000001"),
            ("ms", "2023-06-15 00:00:00", "2023-06-15 00:00:00.001"),
            ("s", "2023-06-15 00:00:00", "2023-06-15 00:00:01"),
            ("min", "2023-06-15 00:00", "2023-06-15 00:01"),
            ("h", "2023-06-15 00:00", "2023-06-15 01:00"),
            ("D", "2023-06-15", "2023-06-16"),
            ("W", "2023-06-12", "2023-06-19"),
            ("SM", "2023-06-01", "2023-06-16"),
            ("M", "2023-06-01", "2023-07-01"),
            ("Q", "2023-04-01", "2023-07-01"),
            ("Y", "2023-01-01", "2024-01-01"),
        ],
        ids=str,
    )
    def test_daily_period_is_read_through_its_start(self, freq, start, end):
        """A daily ``Period`` behaves like its first instant, midnight."""
        result = get_period_boundaries(pd.Period("2023-06-15", freq="D"), freq)

        assert result == (_ts(start), _ts(end))

    def test_period_finer_than_frequency(self):
        """A minute ``Period`` is located by its start in a coarser period."""
        assert get_period_boundaries(pd.Period("2023-06-15 14:35", freq="min"), "h") == (
            _ts("2023-06-15 14:00"),
            _ts("2023-06-15 15:00"),
        )

    def test_period_coarser_than_frequency_uses_its_first_instant(self):
        """A yearly ``Period`` asked in months gives January (documented)."""
        assert get_period_boundaries(pd.Period("2023", freq="Y"), "M") == (_ts("2023-01-01"), _ts("2023-02-01"))

    def test_datetime64_is_accepted(self):
        """A ``numpy.datetime64`` is converted like a ``Timestamp``."""
        assert get_period_start(np.datetime64("2023-06-15T14:35"), "M") == _ts("2023-06-01")

    def test_nat_raises_value_error(self):
        """``NaT`` has no period."""
        with pytest.raises(ValueError, match="NaT"):
            get_period_start(pd.NaT, "M")

    @pytest.mark.parametrize("value", ["2023-06-15", 20230615, None, date_type(2023, 6, 15)],
                             ids=["str", "int", "none", "date-not-datetime"])
    def test_unsupported_type_raises_type_error(self, value):
        """Strings, numbers and plain dates are rejected with a clear message."""
        with pytest.raises(TypeError, match="must be a datetime"):
            get_period_start(value, "M")


class TestInvalidFrequency:
    """An unsupported frequency or anchor is rejected by all three functions."""

    _FUNCTIONS = [get_period_start, get_period_end, get_period_boundaries]

    @pytest.mark.parametrize("function", _FUNCTIONS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("freq", ["invalid_freq", "", "H", "A"], ids=repr)
    def test_unknown_frequency_raises_value_error(self, function, freq):
        """Unknown names and removed pandas aliases raise ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            function(datetime(2023, 6, 15), freq)

    @pytest.mark.parametrize("function", _FUNCTIONS, ids=lambda f: f.__name__)
    def test_non_string_frequency_raises_value_error(self, function):
        """A non-string frequency raises ``ValueError``."""
        with pytest.raises(ValueError, match="must be a string"):
            function(datetime(2023, 6, 15), None)

    @pytest.mark.parametrize(
        "freq, message",
        [
            ("W-XYZ", "Invalid anchor"),
            ("Q-FOO", "Invalid anchor"),
            ("W-JAN", "Invalid anchor"),
            ("Q-MON", "Invalid anchor"),
            ("D-MON", "does not accept an anchor"),
            ("h-JAN", "does not accept an anchor"),
        ],
        ids=str,
    )
    def test_invalid_anchor_raises_value_error(self, freq, message):
        """An anchor of the wrong kind, or on a base without anchor, is rejected."""
        with pytest.raises(ValueError, match=message):
            get_period_start(datetime(2023, 6, 15), freq)
