"""Tests for ``tsforecast.utils.position.converter.PeriodPositionConverter``.

Covers the public methods of ``PeriodPositionConverter``:

- ``get_conversion_factor`` (sign-only factor: ``1.0`` / ``-1.0``);
- ``convert_offset`` (pandas offset string -> offset describing the same
  periods at another position: multiplier kept, month anchor shifted,
  frequencies without a start/end variant unchanged);
- ``convert`` on a bare ``DatetimeIndex``, a ``Series`` / ``DataFrame`` with a
  ``DatetimeIndex``, and a panel (``MultiIndex``, 2 and 3 levels, frequency
  inferred per entity), including the lazily imported helpers of
  ``tsforecast.utils.frequency`` (``detect_index_frequency``,
  ``normalize_frequency``, ``detect_dataset_frequency``).

Central invariants: a position conversion moves each date *within its own
period* to the other bound of that period, on the native pandas grid of the
target offset (``MS`` -> exactly ``pd.date_range(freq='ME')``); it never
merges two distinct dates (a frequency coarser than the data is rejected);
frequencies without a start/end variant in pandas (``D``, ``B``, ``W``,
``h``...) are left unchanged, consistently with ``convert_offset``. Golden
dates are computed by hand (leap February 2024, anchored quarters, fiscal
years).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

import tsforecast.utils.frequency as frequency_package
from tests.support.perturbations import (
    empty_like,
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_period_index,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)
from tsforecast.utils.abc.converter import TemporalConverter
from tsforecast.utils.position.converter import PeriodPositionConverter


# =============================================================================
# Utilitaires
# =============================================================================


def _dates(*values: str) -> pd.DatetimeIndex:
    """Build a ``DatetimeIndex`` from ISO date strings (hand-written golden dates)."""
    return pd.DatetimeIndex(list(values))


def _panel(entities: List[str], dates: List[pd.Timestamp], name: str = "v") -> pd.Series:
    """Build a two-level panel ``Series`` (``entity``, ``date``) valued ``0..n-1``."""
    index = pd.MultiIndex.from_arrays([entities, dates], names=["entity", "date"])
    return pd.Series(np.arange(len(dates), dtype=float), index=index, name=name)


def _date_level(data) -> pd.DatetimeIndex:
    """Return the date level of a panel, or the index of a time series."""
    if isinstance(data.index, pd.MultiIndex):
        return data.index.get_level_values(-1)
    return data.index


@pytest.fixture
def converter() -> PeriodPositionConverter:
    """A fresh ``PeriodPositionConverter`` instance."""
    return PeriodPositionConverter()


@pytest.fixture
def mixed_frequency_panel() -> pd.Series:
    """Two-entity panel with a different frequency per entity.

    Entity ``A`` is quarterly (``QS`` 2023), entity ``B`` monthly (``MS``
    from January to April 2020, a leap year): the frequency must be
    inferred entity by entity.
    """
    dates = list(pd.date_range("2023-01-01", periods=3, freq="QS")) + list(
        pd.date_range("2020-01-01", periods=4, freq="MS")
    )
    return _panel(["A"] * 3 + ["B"] * 4, dates)


# =============================================================================
# Contrat général
# =============================================================================


class TestContract:
    """Place in the ``TemporalConverter`` hierarchy and ``get_conversion_factor``."""

    def test_is_a_temporal_converter(self, converter):
        """The class implements the ``TemporalConverter`` contract."""
        assert isinstance(converter, TemporalConverter)

    @pytest.mark.parametrize(
        "from_unit, to_unit, expected",
        [
            pytest.param("S", "S", 1.0, id="same-code"),
            pytest.param("start", "S", 1.0, id="same-mixed-spelling"),
            pytest.param("end", "end", 1.0, id="same-literal"),
            pytest.param("S", "E", -1.0, id="code-to-code"),
            pytest.param("start", "end", -1.0, id="literal-to-literal"),
            pytest.param("S", "end", -1.0, id="code-to-literal"),
            pytest.param("end", "S", -1.0, id="literal-to-code"),
        ],
    )
    def test_conversion_factor_is_a_sign(self, converter, from_unit, to_unit, expected):
        """No real ratio between positions: ``1.0`` if identical, ``-1.0`` if a shift is needed."""
        assert converter.get_conversion_factor(from_unit, to_unit) == expected

    @pytest.mark.parametrize("from_unit, to_unit", [("foo", "E"), ("S", "foo"), ("s", "E")])
    def test_conversion_factor_rejects_invalid_positions(self, converter, from_unit, to_unit):
        """An unsupported position on either side raises."""
        with pytest.raises(ValueError, match="Unsupported position"):
            converter.get_conversion_factor(from_unit, to_unit)


# =============================================================================
# convert_offset
# =============================================================================


class TestConvertOffset:
    """``convert_offset``: pandas offset string -> same periods at the target position."""

    @pytest.mark.parametrize(
        "offset, to_position, expected",
        [
            # Mensuel : position explicite (S/E) ou implicite (M)
            ("MS", "S", "MS"), ("MS", "E", "ME"),
            ("ME", "S", "MS"), ("ME", "E", "ME"),
            ("M", "S", "MS"), ("M", "E", "ME"),
            # Trimestriel
            ("QS", "S", "QS"), ("QS", "E", "QE"),
            ("QE", "S", "QS"), ("QE", "E", "QE"),
            ("Q", "S", "QS"), ("Q", "E", "QE"),
            # Annuel
            ("YS", "S", "YS"), ("YS", "E", "YE"),
            ("YE", "S", "YS"), ("YE", "E", "YE"),
            ("Y", "S", "YS"), ("Y", "E", "YE"),
            # Semi-mensuel : grilles pandas SMS (1 et 15) / SME (15 et fin de mois)
            ("SMS", "E", "SME"), ("SME", "S", "SMS"), ("SMS", "S", "SMS"),
            ("SMS-15", "E", "SME-15"),
            # Littéraux de position acceptés au même titre que les codes
            ("MS", "end", "ME"), ("QE", "start", "QS"),
        ],
    )
    def test_position_aware_frequencies(self, converter, offset, to_position, expected):
        """Monthly, quarterly, yearly and semi-monthly offsets take the requested position."""
        assert converter.convert_offset(offset, to_position) == expected

    @pytest.mark.parametrize(
        "offset, to_position, expected",
        [
            # Ancres par défaut de pandas : QS-JAN <-> QE-DEC, YS-JAN <-> YE-DEC
            pytest.param("QE-DEC", "S", "QS-JAN", id="QE-DEC-to-start"),
            pytest.param("QE-DEC", "E", "QE-DEC", id="QE-DEC-unchanged"),
            pytest.param("QS-JAN", "E", "QE-DEC", id="QS-JAN-to-end"),
            pytest.param("YE-DEC", "S", "YS-JAN", id="YE-DEC-to-start"),
            # QS-FEB : trimestres févr.-avr., mai-juil., ... -> fins en janvier, avril, ...
            pytest.param("QS-FEB", "E", "QE-JAN", id="QS-FEB-to-end"),
            pytest.param("QS-FEB", "S", "QS-FEB", id="QS-FEB-unchanged"),
            # QE-NOV : trimestres déc.-févr., mars-mai, ... -> débuts en décembre, mars, ...
            pytest.param("QE-NOV", "S", "QS-DEC", id="QE-NOV-to-start"),
            # Exercice juillet-juin
            pytest.param("YS-JUL", "E", "YE-JUN", id="YS-JUL-to-end"),
            pytest.param("YE-JUN", "S", "YS-JUL", id="YE-JUN-to-start"),
            # Ancre sans position : pandas la lit comme mois de fin ('Q-DEC' = 'QE-DEC')
            pytest.param("Q-DEC", "S", "QS-JAN", id="unpositioned-anchor-to-start"),
            pytest.param("2QS-FEB", "E", "2QE-JAN", id="multiplier-and-anchor"),
        ],
    )
    def test_anchor_is_shifted_to_describe_the_same_periods(self, converter, offset, to_position, expected):
        """A month anchor is carried over, shifted so that the periods stay the same.

        ``QE-NOV`` (quarters ending in November, February, May, August) at the
        start position is ``QS-DEC``; ``QS`` (= ``QS-JAN``) would describe
        other quarters.
        """
        assert converter.convert_offset(offset, to_position) == expected

    @pytest.mark.parametrize(
        "offset, to_position, expected",
        [
            pytest.param("2MS", "E", "2ME", id="2MS-to-end"),
            pytest.param("2ME", "S", "2MS", id="2ME-to-start"),
            pytest.param("2MS", "S", "2MS", id="2MS-unchanged"),
            pytest.param("3QS", "end", "3QE", id="3QS-to-end"),
            pytest.param("12ME", "start", "12MS", id="12ME-to-start"),
            pytest.param("2YE", "S", "2YS", id="2YE-to-start"),
            # Un multiplicateur explicite égal à 1 est normalisé (omis)
            pytest.param("1MS", "E", "ME", id="explicit-multiplier-one-omitted"),
        ],
    )
    def test_multiplier_is_kept(self, converter, offset, to_position, expected):
        """A leading multiplier is carried over unchanged ('2MS' -> '2ME')."""
        assert converter.convert_offset(offset, to_position) == expected

    @pytest.mark.parametrize("to_position", ["S", "E"])
    @pytest.mark.parametrize(
        "offset",
        ["D", "5D", "B", "W", "W-SUN", "W-MON", "2W-MON", "h", "2h", "min", "s", "ms"],
    )
    def test_frequency_without_position_is_unchanged(self, converter, offset, to_position):
        """Offsets without a start/end variant in pandas are returned unchanged.

        A week is anchored on a weekday (``W-MON``), not on a position: pandas
        knows neither ``'WS'`` / ``'WE'`` nor ``'BS'`` / ``'BE'``.
        """
        assert converter.convert_offset(offset, to_position) == offset

    @pytest.mark.parametrize("to_position", ["S", "E"])
    @pytest.mark.parametrize(
        "offset",
        [
            "MS", "QE", "Y", "2MS", "12ME", "QS-FEB", "QE-NOV", "YS-JUL", "SMS", "SME",
            "D", "B", "W", "W-MON", "2h",
        ],
    )
    def test_result_is_a_valid_pandas_offset(self, converter, offset, to_position):
        """The returned string is always understood by pandas (``to_offset`` does not raise)."""
        assert to_offset(converter.convert_offset(offset, to_position)) is not None

    @pytest.mark.parametrize(
        "offset",
        [
            # Alias annuels pandas abandonnés délibérément (consolidation parse_frequency)
            pytest.param("A", id="abandoned-alias-A"),
            pytest.param("AS", id="abandoned-alias-AS"),
            pytest.param("AE", id="abandoned-alias-AE"),
            pytest.param("foo", id="unknown-frequency"),
            pytest.param("", id="empty-string"),
            pytest.param("2", id="multiplier-only"),
            pytest.param("QS-XYZ", id="unknown-anchor-month"),
        ],
    )
    def test_unsupported_offset_raises(self, converter, offset):
        """An offset outside the supported frequencies is rejected, never passed through."""
        with pytest.raises(ValueError):
            converter.convert_offset(offset, "E")

    @pytest.mark.parametrize("to_position", ["middle", "s", "End", None])
    def test_invalid_target_position_raises(self, converter, to_position):
        """The target position goes through ``normalize_position``."""
        with pytest.raises(ValueError):
            converter.convert_offset("MS", to_position)


# =============================================================================
# convert : DatetimeIndex nu
# =============================================================================


class TestConvertDatetimeIndex:
    """``convert`` on a bare ``DatetimeIndex``: golden dates computed by hand."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            pytest.param(
                pd.date_range("2023-12-01", periods=4, freq="MS"),
                _dates("2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"),
                id="monthly-leap-february",
            ),
            pytest.param(
                pd.date_range("2023-01-01", periods=3, freq="MS"),
                _dates("2023-01-31", "2023-02-28", "2023-03-31"),
                id="monthly-common-february",
            ),
            pytest.param(
                pd.date_range("2024-01-01", periods=4, freq="QS"),
                _dates("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"),
                id="quarterly",
            ),
            pytest.param(
                pd.date_range("2023-01-01", periods=3, freq="YS"),
                _dates("2023-12-31", "2024-12-31", "2025-12-31"),
                id="yearly",
            ),
        ],
    )
    def test_start_to_end_lands_on_last_day_of_period(self, converter, source, expected):
        """Start -> end moves each date to midnight of the last day of its period."""
        result = converter.convert(source, "start", "end")
        pd.testing.assert_index_equal(result, expected, check_names=False, exact=False)

    @pytest.mark.parametrize(
        "source, expected",
        [
            pytest.param(
                _dates("2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"),
                _dates("2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01"),
                id="monthly-leap-february",
            ),
            pytest.param(
                pd.date_range("2024-03-31", periods=4, freq="QE"),
                _dates("2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"),
                id="quarterly",
            ),
            pytest.param(
                pd.date_range("2023-12-31", periods=3, freq="YE"),
                _dates("2023-01-01", "2024-01-01", "2025-01-01"),
                id="yearly",
            ),
        ],
    )
    def test_end_to_start_lands_on_first_day_of_period(self, converter, source, expected):
        """End -> start moves each date to midnight of the first day of its period."""
        result = converter.convert(source, "end", "start")
        pd.testing.assert_index_equal(result, expected, check_names=False, exact=False)

    @pytest.mark.parametrize("start_freq, end_freq", [("MS", "ME"), ("QS", "QE"), ("YS", "YE"), ("SMS", "SME")])
    def test_start_to_end_matches_pandas_end_grid(self, converter, start_freq, end_freq):
        """Start -> end yields exactly the native pandas end-anchored grid.

        A ``MS`` index converted to end aligns (join, ``reindex``) with a
        native ``ME`` index of the same calendar: midnight of the last day.
        """
        source = pd.date_range("2024-01-01", periods=4, freq=start_freq)
        expected = pd.date_range("2024-01-01", periods=4, freq=end_freq)
        assert list(converter.convert(source, "S", "E")) == list(expected)

    @pytest.mark.parametrize("freq", ["ME", "QE", "YE", "SME", "2ME", "QE-NOV", "YE-JUN"])
    def test_end_start_end_roundtrip_is_identity(self, converter, freq):
        """End -> start -> end gives back a native end-anchored index exactly."""
        source = pd.date_range("2023-01-01", periods=6, freq=freq)
        roundtrip = converter.convert(converter.convert(source, "E", "S", freq=freq), "S", "E", freq=freq)
        assert list(roundtrip) == list(source)

    @pytest.mark.parametrize("freq", ["MS", "QS", "YS", "SMS", "2MS", "QS-FEB", "YS-JUL"])
    def test_start_end_start_roundtrip_is_identity(self, converter, freq):
        """Start -> end -> start gives back the original dates exactly."""
        source = pd.date_range("2022-01-01", periods=6, freq=freq)
        roundtrip = converter.convert(converter.convert(source, "S", "E", freq=freq), "E", "S", freq=freq)
        assert list(roundtrip) == list(source)

    @pytest.mark.parametrize(
        "start_freq",
        ["MS", "QS", "YS", "SMS", "2MS", "3QS", "QS-FEB", "QS-DEC", "YS-JUL", "2QS-FEB"],
    )
    def test_converted_index_follows_convert_offset(self, converter, start_freq):
        """Property: converting a ``start_freq`` grid gives the grid of ``convert_offset(start_freq, 'E')``.

        Link between both methods: the converted offset describes exactly the
        converted index (same anchor, same multiplier).
        """
        source = pd.date_range("2023-01-01", periods=5, freq=start_freq)
        result = converter.convert(source, "S", "E", freq=start_freq)
        end_freq = converter.convert_offset(start_freq, "E")
        expected = pd.date_range(result[0], periods=5, freq=end_freq)
        assert list(result) == list(expected)

    @pytest.mark.parametrize(
        "source, from_pos, to_pos, expected",
        [
            pytest.param(
                # Trimestres févr.-avr., mai-juil., août-oct. : fins au 30/04, 31/07, 31/10
                pd.date_range("2024-02-01", periods=3, freq="QS-FEB"), "S", "E",
                _dates("2024-04-30", "2024-07-31", "2024-10-31"),
                id="QS-FEB-to-end",
            ),
            pytest.param(
                # Trimestres sept.-nov., déc.-févr. (29/02 bissextile), mars-mai
                pd.date_range("2023-11-30", periods=3, freq="QE-NOV"), "E", "S",
                _dates("2023-09-01", "2023-12-01", "2024-03-01"),
                id="QE-NOV-to-start",
            ),
            pytest.param(
                # Exercices juillet-juin (3 dates : pandas n'infère rien sur 2)
                pd.date_range("2023-07-01", periods=3, freq="YS-JUL"), "S", "E",
                _dates("2024-06-30", "2025-06-30", "2026-06-30"),
                id="YS-JUL-to-end",
            ),
        ],
    )
    def test_anchored_frequency_uses_its_own_periods(self, converter, source, from_pos, to_pos, expected):
        """Anchored frequencies convert within their own periods, not calendar ones (detected freq)."""
        result = converter.convert(source, from_pos, to_pos)
        pd.testing.assert_index_equal(result, expected, check_names=False, exact=False)

    @pytest.mark.parametrize(
        "source, from_pos, to_pos, expected",
        [
            pytest.param(
                # Blocs de deux mois janv.-févr., mars-avr., mai-juin
                pd.date_range("2024-01-01", periods=3, freq="2MS"), "S", "E",
                _dates("2024-02-29", "2024-04-30", "2024-06-30"),
                id="2MS-to-end",
            ),
            pytest.param(
                pd.date_range("2024-02-29", periods=3, freq="2ME"), "E", "S",
                _dates("2024-01-01", "2024-03-01", "2024-05-01"),
                id="2ME-to-start",
            ),
            pytest.param(
                # Semestres civils
                pd.date_range("2024-01-01", periods=3, freq="2QS"), "S", "E",
                _dates("2024-06-30", "2024-12-31", "2025-06-30"),
                id="2QS-to-end",
            ),
        ],
    )
    def test_multiplied_frequency(self, converter, source, from_pos, to_pos, expected):
        """A multiplied frequency (detected from the index) converts whole multi-period blocks."""
        result = converter.convert(source, from_pos, to_pos)
        pd.testing.assert_index_equal(result, expected, check_names=False, exact=False)

    @pytest.mark.parametrize(
        "source, from_pos, to_pos, expected",
        [
            pytest.param(
                pd.date_range("2024-01-01", periods=4, freq="SMS"), "S", "E",
                _dates("2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"),
                id="SMS-to-end",
            ),
            pytest.param(
                pd.date_range("2024-01-15", periods=4, freq="SME"), "E", "S",
                _dates("2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"),
                id="SME-to-start",
            ),
        ],
    )
    def test_semi_monthly_pairs_native_grids(self, converter, source, from_pos, to_pos, expected):
        """Semi-monthly dates pair by rank in the month: 1st <-> 15th, 15th <-> month end."""
        result = converter.convert(source, from_pos, to_pos)
        pd.testing.assert_index_equal(result, expected, check_names=False, exact=False)

    @pytest.mark.parametrize("freq", ["2SMS", "SMS-10"])
    def test_unsupported_semi_monthly_variant_raises(self, converter, freq):
        """Multiplied or non-default semi-monthly frequencies are rejected with a dedicated message."""
        source = pd.date_range("2024-01-01", periods=4, freq="SMS")
        with pytest.raises(ValueError, match="Unsupported semi-monthly frequency"):
            converter.convert(source, "S", "E", freq=freq)

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(pd.date_range("2024-01-01", periods=5, freq="D"), id="daily"),
            pytest.param(pd.bdate_range("2024-01-01", periods=5), id="business-daily"),
            pytest.param(pd.date_range("2024-01-07", periods=4, freq="W-SUN"), id="weekly-sunday"),
            pytest.param(pd.date_range("2024-01-03", periods=4, freq="W-WED"), id="weekly-wednesday"),
            pytest.param(pd.date_range("2024-01-01 06:00", periods=5, freq="h"), id="hourly"),
        ],
    )
    @pytest.mark.parametrize("from_pos, to_pos", [("S", "E"), ("E", "S")])
    def test_frequency_without_position_is_identity(self, converter, source, from_pos, to_pos):
        """Frequencies without a start/end variant are left unchanged, as in ``convert_offset``."""
        assert list(converter.convert(source, from_pos, to_pos)) == list(source)

    @pytest.mark.parametrize("freq", [None, "M", "MS", "ME", "monthly"])
    def test_frequency_spelling_does_not_matter(self, converter, freq):
        """Detected, bare, positioned or literal monthly frequency give the same result."""
        source = pd.date_range("2024-01-01", periods=3, freq="MS")
        result = converter.convert(source, "S", "E", freq=freq)
        pd.testing.assert_index_equal(
            result, _dates("2024-01-31", "2024-02-29", "2024-03-31"), check_names=False, exact=False
        )

    @pytest.mark.parametrize("freq", ["MS", "QS", "YS"])
    def test_dates_stay_in_their_period(self, converter, freq):
        """Property: every converted date belongs to the period of its source date."""
        source = pd.date_range("2022-01-01", periods=12, freq=freq)
        result = converter.convert(source, "S", "E")
        period_freq = freq[0]
        assert (result.to_period(period_freq) == source.to_period(period_freq)).all()

    def test_off_grid_dates_move_to_their_period_bound(self, converter):
        """Dates inside a period (not on its start) still move to that period's end."""
        source = _dates("2024-01-10", "2024-02-20", "2024-03-01")
        result = converter.convert(source, "S", "E", freq="M")
        pd.testing.assert_index_equal(result, _dates("2024-01-31", "2024-02-29", "2024-03-31"))

    def test_time_of_day_is_dropped(self, converter):
        """Converted dates are period bounds at midnight, whatever the source time of day."""
        source = _dates("2024-01-01 10:30", "2024-02-01 23:59")
        result = converter.convert(source, "S", "E", freq="M")
        pd.testing.assert_index_equal(result, _dates("2024-01-31", "2024-02-29"))

    def test_time_zone_is_kept_across_dst(self, converter):
        """A tz-aware index keeps its time zone; bounds stay at local midnight across DST.

        Paris switches to summer time on 2024-03-31: the end of March stays at
        local midnight (not 23:00 the day before), the end of April is at
        +02:00.
        """
        source = pd.date_range("2024-02-01", periods=3, freq="MS", tz="Europe/Paris")
        result = converter.convert(source, "S", "E")
        expected = pd.DatetimeIndex(["2024-02-29", "2024-03-31", "2024-04-30"]).tz_localize("Europe/Paris")
        pd.testing.assert_index_equal(result, expected, exact=False)

    def test_index_name_is_kept(self, converter):
        """The index name survives the conversion."""
        source = pd.date_range("2024-01-01", periods=3, freq="MS", name="période")
        assert converter.convert(source, "S", "E").name == "période"

    def test_same_position_returns_the_input_object(self, converter):
        """Identical positions short-circuit: the very same object is returned (no copy)."""
        source = pd.date_range("2024-01-01", periods=3, freq="MS")
        assert converter.convert(source, "S", "start") is source

    def test_unsorted_index_is_converted_element_wise(self, converter):
        """An unsorted index keeps its order: each date is converted in place.

        The frequency is detected on the sorted index (``FrequencyDetector``
        fallback), then dates are converted element-wise.
        """
        source = _dates("2024-03-01", "2024-01-01", "2024-04-01", "2024-02-01")
        result = converter.convert(source, "S", "E")
        pd.testing.assert_index_equal(
            result, _dates("2024-03-31", "2024-01-31", "2024-04-30", "2024-02-29"), check_names=False
        )

    def test_short_index_with_duplicates_is_converted(self, converter):
        """A short monthly index with one duplicated date is still converted to month ends.

        Spacings between unique dates: 31 then 29 days -> monthly. Before the
        fix (ANO-UTILS-014), the zero spacing of the duplicate won as modal
        spacing and the detected frequency was ``'ns'`` (silent no-op).
        """
        source = _dates("2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01")
        result = converter.convert(source, "S", "E")
        assert list(result) == list(_dates("2024-01-31", "2024-01-31", "2024-02-29", "2024-03-31"))

    def test_single_date_with_explicit_frequency(self, converter):
        """One date is enough when the frequency is given."""
        result = converter.convert(_dates("2024-02-01"), "S", "E", freq="M")
        assert result[0] == pd.Timestamp("2024-02-29")

    def test_single_date_without_frequency_raises(self, converter):
        """No frequency can be inferred from a single date."""
        with pytest.raises(ValueError):
            converter.convert(_dates("2024-02-01"), "S", "E")

    def test_empty_index_with_explicit_frequency(self, converter):
        """An empty index converts to an empty ``DatetimeIndex``."""
        result = converter.convert(pd.DatetimeIndex([]), "S", "E", freq="M")
        assert isinstance(result, pd.DatetimeIndex) and len(result) == 0

    def test_empty_index_without_frequency_raises(self, converter):
        """No frequency can be inferred from an empty index."""
        with pytest.raises(ValueError):
            converter.convert(pd.DatetimeIndex([]), "S", "E")

    def test_undetectable_frequency_raises_dedicated_error(self, converter):
        """Irregular spacing (45 then 50 days) matches no known frequency: explicit error."""
        source = _dates("2023-01-01", "2023-02-15", "2023-04-06")
        with pytest.raises(ValueError, match="Cannot infer frequency from data"):
            converter.convert(source, "S", "E")

    @pytest.mark.parametrize("freq", ["Q", "Y"])
    def test_coarser_frequency_is_rejected(self, converter, freq):
        """A ``freq`` coarser than the data would merge distinct dates: explicit error.

        ``freq='Q'`` on monthly dates would send January, February and March
        to 03-31: January and February would leave their month
        (ANO-UTILS-010). Aggregating to a lower frequency belongs to
        ``groupby`` / ``resample``, not to a position change.
        """
        source = pd.date_range("2024-01-01", periods=3, freq="MS")
        with pytest.raises(ValueError, match="coarser than the data"):
            converter.convert(source, "S", "E", freq=freq)


# =============================================================================
# convert : Series / DataFrame à DatetimeIndex
# =============================================================================


class TestConvertTimeSeries:
    """``convert`` on a ``Series`` / ``DataFrame`` with a ``DatetimeIndex``."""

    @pytest.fixture
    def monthly_series(self) -> pd.Series:
        """Monthly start-anchored series over the leap February 2024."""
        index = pd.date_range("2024-01-01", periods=3, freq="MS", name="date")
        return pd.Series([10.0, 20.0, 30.0], index=index, name="x")

    def test_series_golden_dates_and_values(self, converter, monthly_series):
        """Index moved to month ends, values and name unchanged."""
        result = converter.convert(monthly_series, "start", "end")
        expected = pd.Series(
            [10.0, 20.0, 30.0], index=_dates("2024-01-31", "2024-02-29", "2024-03-31"), name="x"
        )
        expected.index.name = "date"
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_dataframe_keeps_columns_values_and_index_name(self, converter, monthly_series):
        """Every column shares the new index; column labels and values unchanged."""
        frame = pd.DataFrame({"a": monthly_series, "b": monthly_series * 2})
        result = converter.convert(frame, "S", "E")
        pd.testing.assert_frame_equal(
            result,
            frame.set_axis(_dates("2024-01-31", "2024-02-29", "2024-03-31").rename("date")),
            check_freq=False,
        )

    def test_dataframe_keeps_mixed_dtypes(self, converter, monthly_series):
        """Integer, float, boolean and string columns keep their dtype.

        Before the fix (ANO-UTILS-012), the result was rebuilt from
        ``.values`` and every column became ``object``.
        """
        frame = pd.DataFrame(
            {"entier": [1, 2, 3], "réel": [1.5, 2.5, 3.5], "booléen": [True, False, True], "texte": list("abc")},
            index=monthly_series.index,
        )
        result = converter.convert(frame, "S", "E")
        pd.testing.assert_series_equal(result.dtypes, frame.dtypes)

    def test_input_is_not_modified(self, converter, monthly_series):
        """The conversion works on a copy: the input index is untouched."""
        original_index = monthly_series.index.copy()
        converter.convert(monthly_series, "S", "E")
        pd.testing.assert_index_equal(monthly_series.index, original_index)

    def test_special_column_names_are_kept(self, converter, monthly_series):
        """Column names with spaces, accents and symbols are untouched."""
        frame, mapping = with_special_column_names(pd.DataFrame({"a": monthly_series, "b": monthly_series}))
        result = converter.convert(frame, "S", "E")
        assert list(result.columns) == list(mapping.values())

    def test_same_position_returns_the_input_object(self, converter, monthly_series):
        """Identical positions short-circuit: the very same object is returned (no copy)."""
        assert converter.convert(monthly_series, "end", "E") is monthly_series

    def test_unsorted_series_keeps_date_value_pairs(self, converter):
        """A shuffled series is converted in place: each value stays with its own date."""
        index = pd.date_range("2023-01-01", periods=12, freq="MS", name="date")
        series = pd.Series(np.arange(12.0), index=index, name="x")
        shuffled = shuffle_rows(series, seed=3)
        result = converter.convert(shuffled, "S", "E")
        # Valeur d'or : la valeur i est associée à la fin du mois i+1 de 2023
        expected = converter.convert(series, "S", "E").reindex(result.index)
        pd.testing.assert_series_equal(result, expected)

    def test_duplicated_rows_are_kept(self, converter):
        """Duplicated dates are converted like the others and remain duplicated."""
        index = pd.date_range("2023-01-01", periods=6, freq="MS", name="date")
        duplicated = with_duplicated_rows(pd.Series(np.arange(6.0), index=index, name="x"), n=1)
        result = converter.convert(duplicated, "S", "E")
        assert list(result.index[[0, -1]]) == [pd.Timestamp("2023-01-31")] * 2

    def test_undetectable_frequency_raises(self, converter):
        """An irregular series without ``freq`` raises the dedicated error."""
        series = pd.Series([1.0, 2.0, 3.0], index=_dates("2023-01-01", "2023-02-15", "2023-04-06"))
        with pytest.raises(ValueError, match="Cannot infer frequency from data"):
            converter.convert(series, "S", "E")

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(pd.Series([1, 2, 3]), id="range-index"),
            pytest.param(
                pd.Series([1, 2, 3], index=pd.period_range("2024-01", periods=3, freq="M")),
                id="period-index",
            ),
            pytest.param(
                pd.Series([1, 2], index=pd.MultiIndex.from_arrays([_dates("2024-01-01", "2024-02-01")])),
                id="single-level-multiindex",
            ),
        ],
    )
    def test_non_datetime_index_raises(self, converter, value):
        """Only a ``DatetimeIndex`` (or a panel ``MultiIndex``) is supported."""
        with pytest.raises(ValueError, match="Data must have DatetimeIndex or MultiIndex"):
            converter.convert(value, "S", "E")

    @pytest.mark.parametrize("value", [[1, 2, 3], np.arange(3), "2024-01-01"], ids=["list", "array", "str"])
    def test_unsupported_value_type_raises(self, converter, value):
        """Anything else than a ``DatetimeIndex`` / ``Series`` / ``DataFrame`` is rejected."""
        with pytest.raises(ValueError, match="Unsupported value type"):
            converter.convert(value, "S", "E")

    def test_invalid_position_raises_before_any_conversion(self, converter, monthly_series):
        """Positions are validated first, whatever the data."""
        with pytest.raises(ValueError, match="Unsupported position"):
            converter.convert(monthly_series, "middle", "E")

    def test_coarser_explicit_frequency_is_rejected(self, converter, monthly_series):
        """A ``freq`` coarser than the data is rejected instead of merging observations."""
        with pytest.raises(ValueError, match="coarser than the data"):
            converter.convert(monthly_series, "S", "E", freq="Q")


# =============================================================================
# convert : panels (MultiIndex)
# =============================================================================


class TestConvertPanel:
    """``convert`` on a panel: frequency inferred per entity, structure validated."""

    def test_mixed_frequency_panel_golden_dates(self, converter, mixed_frequency_panel):
        """Each entity is converted with its own detected frequency.

        Golden values: A (quarterly 2023) -> 03-31, 06-30, 09-30;
        B (monthly 2020, leap year) -> 01-31, 02-29, 03-31, 04-30.
        """
        result = converter.convert(mixed_frequency_panel, "start", "end")
        expected_dates = _dates(
            "2023-03-31", "2023-06-30", "2023-09-30",
            "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30",
        )
        pd.testing.assert_index_equal(_date_level(result), expected_dates, check_names=False)

    def test_entities_values_and_names_are_kept(self, converter, mixed_frequency_panel):
        """Entity level, values, series name and index names are unchanged."""
        result = converter.convert(mixed_frequency_panel, "S", "E")
        assert (
            list(result.index.get_level_values("entity")),
            list(result.values),
            result.name,
            list(result.index.names),
        ) == (
            list(mixed_frequency_panel.index.get_level_values("entity")),
            list(mixed_frequency_panel.values),
            "v",
            ["entity", "date"],
        )

    def test_dataframe_panel(self, converter, mixed_frequency_panel):
        """A panel ``DataFrame`` gets the same dates as the equivalent ``Series``."""
        frame = mixed_frequency_panel.to_frame()
        result = converter.convert(frame, "S", "E")
        pd.testing.assert_index_equal(
            result.index, converter.convert(mixed_frequency_panel, "S", "E").index
        )

    def test_dataframe_panel_keeps_mixed_dtypes(self, converter, mixed_frequency_panel):
        """Panel columns keep their dtype (integer, float, string)."""
        frame = pd.DataFrame(
            {"entier": np.arange(7), "réel": np.arange(7) * 0.5, "texte": list("abcdefg")},
            index=mixed_frequency_panel.index,
        )
        result = converter.convert(frame, "S", "E")
        pd.testing.assert_series_equal(result.dtypes, frame.dtypes)

    def test_start_end_start_roundtrip_is_identity(self, converter, mixed_frequency_panel):
        """Start -> end -> start gives back the original panel exactly."""
        roundtrip = converter.convert(converter.convert(mixed_frequency_panel, "S", "E"), "E", "S")
        pd.testing.assert_series_equal(roundtrip, mixed_frequency_panel, check_index_type=False)

    def test_three_level_index_golden_dates(self, converter):
        """Grouping is done on every level but the last (``region``, ``entity``)."""
        dates = list(pd.date_range("2023-01-01", periods=3, freq="QS")) * 2 + list(
            pd.date_range("2023-01-01", periods=3, freq="MS")
        )
        index = pd.MultiIndex.from_arrays(
            [["EU"] * 6 + ["NA"] * 3, ["FR"] * 3 + ["DE"] * 3 + ["US"] * 3, dates],
            names=["region", "entity", "date"],
        )
        panel = pd.Series(np.arange(9.0), index=index)
        result = converter.convert(panel, "S", "E")
        # Valeurs d'or : FR et DE trimestriels, US mensuel (fréquence propre au groupe)
        expected = _dates(
            "2023-03-31", "2023-06-30", "2023-09-30",
            "2023-03-31", "2023-06-30", "2023-09-30",
            "2023-01-31", "2023-02-28", "2023-03-31",
        )
        assert (
            list(result.index.names),
            list(result.index.droplevel(-1)),
            list(_date_level(result)),
        ) == (list(index.names), list(index.droplevel(-1)), list(expected))

    def test_reversed_entity_order_is_kept(self, converter, mixed_frequency_panel):
        """Entities are processed in their order of appearance, not sorted."""
        reversed_panel = reverse_entities(mixed_frequency_panel)
        result = converter.convert(reversed_panel, "S", "E")
        assert list(result.index.get_level_values("entity")) == ["B"] * 4 + ["A"] * 3

    def test_non_standard_index_names_are_kept(self, converter, mixed_frequency_panel):
        """Level names are carried over, whatever they are."""
        renamed = with_index_names(mixed_frequency_panel, ["pays / zone", "période"])
        assert list(converter.convert(renamed, "S", "E").index.names) == ["pays / zone", "période"]

    def test_interleaved_entities_raise(self, converter):
        """Entities must form contiguous blocks."""
        panel = _panel(["A", "B", "A", "B"], list(pd.date_range("2023-01-01", periods=4, freq="QS")))
        with pytest.raises(ValueError, match="must be grouped"):
            converter.convert(panel, "S", "E", freq="Q")

    def test_unsorted_dates_within_entity_raise(self, converter):
        """Dates must be increasing within each entity."""
        panel = _panel(["A"] * 3, list(_dates("2023-07-01", "2023-01-01", "2023-04-01")))
        with pytest.raises(ValueError, match="must be sorted within each entity"):
            converter.convert(panel, "S", "E", freq="Q")

    def test_shuffled_panel_raises(self, converter, mixed_frequency_panel):
        """A fully shuffled panel is rejected rather than silently mis-grouped."""
        with pytest.raises(ValueError, match="grouped|sorted"):
            converter.convert(shuffle_rows(mixed_frequency_panel, seed=1), "S", "E")

    @pytest.mark.parametrize(
        "last_level",
        [
            pytest.param([1, 2], id="integer"),
            pytest.param(pd.period_range("2024-01", periods=2, freq="M"), id="period"),
        ],
    )
    def test_non_datetime_last_level_raises(self, converter, last_level):
        """The last level of the ``MultiIndex`` must be a ``DatetimeIndex``."""
        index = pd.MultiIndex.from_arrays([["A", "A"], last_level])
        with pytest.raises(ValueError, match="Last level of MultiIndex must be DatetimeIndex"):
            converter.convert(pd.Series([1.0, 2.0], index=index), "S", "E")

    def test_period_level_from_perturbation_raises(self, converter, mixed_freq_panel):
        """A realistic panel whose date level became a ``PeriodIndex`` is rejected."""
        with pytest.raises(ValueError, match="Last level of MultiIndex must be DatetimeIndex"):
            converter.convert(to_period_index(mixed_freq_panel), "S", "E")

    def test_single_observation_entity_with_explicit_frequency(self, converter):
        """An entity with one date is converted when ``freq`` is given."""
        dates = list(pd.date_range("2024-01-01", periods=3, freq="MS")) + [pd.Timestamp("2024-02-01")]
        panel = _panel(["A"] * 3 + ["B"], dates)
        result = converter.convert(panel, "S", "E", freq="M")
        assert _date_level(result)[-1] == pd.Timestamp("2024-02-29")

    def test_single_observation_entity_without_frequency_raises(self, converter):
        """No frequency can be inferred for an entity reduced to one date: entity-specific error."""
        dates = list(pd.date_range("2024-01-01", periods=3, freq="MS")) + [pd.Timestamp("2024-02-01")]
        with pytest.raises(ValueError, match="Cannot infer frequency for entity B"):
            converter.convert(_panel(["A"] * 3 + ["B"], dates), "S", "E")

    def test_single_observation_panel_with_explicit_frequency(self, converter, mixed_frequency_panel):
        """A panel reduced to its first row is converted when ``freq`` is given."""
        result = converter.convert(single_observation(mixed_frequency_panel), "S", "E", freq="Q")
        assert _date_level(result)[0] == pd.Timestamp("2023-03-31")

    def test_empty_panel_with_explicit_frequency(self, converter, mixed_frequency_panel):
        """An empty panel converts to an empty panel with the same structure.

        Before the fix (ANO-UTILS-013), ``pd.concat([])`` raised
        ``No objects to concatenate``.
        """
        result = converter.convert(empty_like(mixed_frequency_panel), "S", "E", freq="Q")
        assert (len(result), list(result.index.names)) == (0, ["entity", "date"])

    def test_same_position_returns_the_input_object(self, converter, mixed_frequency_panel):
        """Identical positions short-circuit: the very same object is returned (no copy)."""
        assert converter.convert(mixed_frequency_panel, "S", "S") is mixed_frequency_panel

    def test_explicit_frequency_incompatible_with_an_entity_is_rejected(self, converter, mixed_frequency_panel):
        """An explicit ``freq`` applied to every entity must not merge an entity's observations.

        ``freq='Q'`` forced on entity B (monthly) would send January, February
        and March 2020 to 2020-03-31 (ANO-UTILS-010).
        """
        with pytest.raises(ValueError, match="coarser than the data"):
            converter.convert(mixed_frequency_panel, "S", "E", freq="Q")

    def test_undetectable_entity_frequency_raises_dedicated_error(self, converter):
        """An entity with no detectable frequency raises the entity-specific error.

        Entity B is spaced by 45 then 50 days: neither its index nor its
        columns reveal a frequency. Before the fix (ANO-UTILS-011), the
        fallback returned a ``dict`` and the error was about its type.
        """
        dates = list(pd.date_range("2024-01-01", periods=3, freq="MS")) + list(
            _dates("2023-01-01", "2023-02-15", "2023-04-06")
        )
        panel = _panel(["A"] * 3 + ["B"] * 3, dates).to_frame()
        with pytest.raises(ValueError, match="Cannot infer frequency for entity B"):
            converter.convert(panel, "S", "E")

    def test_column_frequency_fallback(self, converter):
        """When an entity's dates are irregular, the finest column frequency is used.

        Entity A: column ``m`` monthly (January to March), column ``x``
        observed on 04-15 and 05-30. Index spacings are 31, 28, 45, 45 days,
        modal spacing 45 -> no index frequency. Column ``m`` (NaN dropped) is
        detected ``MS``: monthly conversion, possible since no two dates share
        a month.
        """
        index = pd.MultiIndex.from_arrays(
            [["A"] * 5, _dates("2023-01-01", "2023-02-01", "2023-03-01", "2023-04-15", "2023-05-30")],
            names=["entity", "date"],
        )
        frame = pd.DataFrame(
            {"m": [1.0, 2.0, 3.0, np.nan, np.nan], "x": [np.nan, np.nan, np.nan, 4.0, 5.0]}, index=index
        )
        assert frequency_package.detect_index_frequency(index.get_level_values(-1), return_format="full") is None
        result = converter.convert(frame, "S", "E")
        assert list(_date_level(result)) == list(
            _dates("2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30", "2023-05-31")
        )


# =============================================================================
# Jeux réalistes (notebook 3)
# =============================================================================


class TestRealisticDatasets:
    """Properties on the realistic notebook-3 datasets (start-anchored, irregular index)."""

    @staticmethod
    def _assert_same_month(original: pd.DatetimeIndex, converted: pd.DatetimeIndex) -> None:
        """Assert that each converted date lies in the month of its source date."""
        assert (converted.to_period("M") == original.to_period("M")).all()

    def test_irregular_timeseries_stays_in_period(self, converter, irregular_index_timeseries):
        """Every row, annual anchors before the monthly grid included, stays in its month.

        The frequency detected on the irregular index is monthly (modal
        spacing): the 2015-2017 annual anchors stay in their month, hence in
        their year.
        """
        result = converter.convert(irregular_index_timeseries, "S", "E")
        self._assert_same_month(irregular_index_timeseries.index, result.index)

    def test_irregular_timeseries_lands_on_month_ends(self, converter, irregular_index_timeseries):
        """Every converted date is a month end at midnight."""
        result = converter.convert(irregular_index_timeseries, "S", "E")
        assert result.index.is_month_end.all() and (result.index == result.index.normalize()).all()

    def test_irregular_timeseries_values_untouched(self, converter, irregular_index_timeseries):
        """Values, NaN pattern, dtypes and columns are unchanged, row by row."""
        result = converter.convert(irregular_index_timeseries, "S", "E")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), irregular_index_timeseries.reset_index(drop=True)
        )

    def test_irregular_timeseries_roundtrip(self, converter, irregular_index_timeseries):
        """Start -> end -> start restores the original index."""
        roundtrip = converter.convert(converter.convert(irregular_index_timeseries, "S", "E"), "E", "S")
        pd.testing.assert_index_equal(roundtrip.index, irregular_index_timeseries.index, exact=False, check_names=False)

    def test_heterogeneous_panel_stays_in_period(self, converter, heterogeneous_coverage_panel):
        """Each entity (own coverage, own irregular index) keeps every row in its month."""
        result = converter.convert(heterogeneous_coverage_panel, "S", "E")
        self._assert_same_month(_date_level(heterogeneous_coverage_panel), _date_level(result))

    def test_heterogeneous_panel_structure_untouched(self, converter, heterogeneous_coverage_panel):
        """Entity sequence, index names, columns and values are unchanged."""
        result = converter.convert(heterogeneous_coverage_panel, "S", "E")
        assert list(result.index.get_level_values("country")) == list(
            heterogeneous_coverage_panel.index.get_level_values("country")
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), heterogeneous_coverage_panel.reset_index(drop=True)
        )

    def test_heterogeneous_panel_roundtrip(self, converter, heterogeneous_coverage_panel):
        """Start -> end -> start restores the original panel."""
        roundtrip = converter.convert(converter.convert(heterogeneous_coverage_panel, "S", "E"), "E", "S")
        pd.testing.assert_frame_equal(roundtrip, heterogeneous_coverage_panel, check_index_type=False)

    def test_three_level_heterogeneous_panel(self, converter, heterogeneous_coverage_panel):
        """A third outer level does not change the per-entity conversion."""
        three_level = to_three_level_index(
            heterogeneous_coverage_panel, region_by_entity={"France": "Ouest", "Allemagne": "Centre"}
        )
        result = converter.convert(three_level, "S", "E")
        two_level = converter.convert(heterogeneous_coverage_panel, "S", "E")
        pd.testing.assert_index_equal(
            result.index.droplevel("region"), two_level.index
        )


# =============================================================================
# Imports différés de tsforecast.utils.frequency
# =============================================================================


class TestDeferredFrequencyImports:
    """The lazily imported helpers of ``tsforecast.utils.frequency`` are actually used.

    ``converter.py`` imports ``detect_index_frequency``,
    ``normalize_frequency`` and ``detect_dataset_frequency`` inside its
    methods (import cycle with ``tsforecast.utils.frequency``). The package
    attribute is therefore resolved at call time: replacing it with a test
    double (``monkeypatch``) proves that each path really goes through it.
    """

    @staticmethod
    def _spy(monkeypatch, name: str) -> list:
        """Wrap ``tsforecast.utils.frequency.<name>`` and record its keyword arguments."""
        calls: list = []
        original = getattr(frequency_package, name)

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(frequency_package, name, spy)
        return calls

    def test_convert_detects_index_frequency_in_full_format(self, converter, monkeypatch):
        """``convert`` without ``freq`` calls ``detect_index_frequency(return_format='full')``."""
        calls = self._spy(monkeypatch, "detect_index_frequency")
        converter.convert(pd.date_range("2024-01-01", periods=3, freq="MS"), "S", "E")
        assert [call["return_format"] for call in calls] == ["full"]

    def test_explicit_frequency_skips_detection(self, converter, monkeypatch):
        """An explicit ``freq`` bypasses frequency detection entirely."""
        calls = self._spy(monkeypatch, "detect_index_frequency")
        converter.convert(pd.date_range("2024-01-01", periods=3, freq="MS"), "S", "E", freq="M")
        assert calls == []

    def test_index_conversion_decomposes_the_frequency(self, converter, monkeypatch):
        """The base code is validated by ``normalize_frequency``; multiplier, position and anchor come from ``parse_frequency``."""
        calls = self._spy(monkeypatch, "normalize_frequency")
        converter.convert(pd.date_range("2024-02-01", periods=3, freq="2QS-FEB"), "S", "E", freq="2QS-FEB")
        assert calls == [{"frequency": "Q"}]

    def test_panel_detects_frequency_once_per_entity(self, converter, monkeypatch, mixed_frequency_panel):
        """A panel without ``freq`` runs one detection per entity group."""
        calls = self._spy(monkeypatch, "detect_index_frequency")
        converter.convert(mixed_frequency_panel, "S", "E")
        assert len(calls) == 2

    def test_panel_falls_back_on_dataset_detection(self, converter, monkeypatch, mixed_frequency_panel):
        """When index detection fails, ``detect_dataset_frequency`` provides one consistent frequency.

        Test doubles: index detection fails (``None``) and column detection
        returns ``'MS'``. The fallback must request a single frequency
        (``check_consistency=True``, the finest one) in full format.
        """
        fallback_calls: list = []
        monkeypatch.setattr(frequency_package, "detect_index_frequency", lambda **kwargs: None)

        def fake_dataset_detection(**kwargs):
            fallback_calls.append(kwargs)
            return "MS"

        monkeypatch.setattr(frequency_package, "detect_dataset_frequency", fake_dataset_detection)
        result = converter.convert(mixed_frequency_panel, "S", "E")
        options = {key: fallback_calls[0][key] for key in ("return_format", "check_consistency", "consistency_mode")}
        assert (len(fallback_calls), options, _date_level(result)[-1]) == (
            2,
            {"return_format": "full", "check_consistency": True, "consistency_mode": "highest"},
            pd.Timestamp("2020-04-30"),
        )

    def test_panel_raises_when_both_detections_fail(self, converter, monkeypatch, mixed_frequency_panel):
        """If both detections return ``None``, the entity-specific error is raised."""
        monkeypatch.setattr(frequency_package, "detect_index_frequency", lambda **kwargs: None)
        monkeypatch.setattr(frequency_package, "detect_dataset_frequency", lambda **kwargs: None)
        with pytest.raises(ValueError, match="Cannot infer frequency for entity A"):
            converter.convert(mixed_frequency_panel, "S", "E")


# =============================================================================
# Branches défensives des méthodes privées
# =============================================================================


@pytest.mark.internal
class TestPrivateDefensiveBranches:
    """Defensive branches of the private helpers, unreachable through ``convert``.

    ``convert`` short-circuits identical positions and dispatches on the
    index type before calling these methods: their guards are only
    reachable through a direct call.
    """

    def test_datetime_index_same_position_returns_input(self, converter):
        """``_convert_datetime_index`` returns the index unchanged for identical positions."""
        index = pd.date_range("2024-01-01", periods=3, freq="MS")
        assert converter._convert_datetime_index(index, "S", "S", "M") is index

    def test_time_series_requires_datetime_index(self, converter):
        """``_convert_time_series`` rejects a non-datetime index."""
        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            converter._convert_time_series(pd.Series([1, 2]), "S", "E", "M")

    def test_panel_requires_multiindex(self, converter):
        """``_convert_panel`` rejects a non-``MultiIndex`` input."""
        series = pd.Series([1, 2], index=pd.date_range("2024-01-01", periods=2, freq="MS"))
        with pytest.raises(ValueError, match="Panel data must have a MultiIndex"):
            converter._convert_panel(series, "S", "E", "M")
