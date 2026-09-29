"""Tests for ``tsforecast.utils.duration.converter.DurationConverter``.

Covers the two public methods, ``convert`` and ``get_conversion_factor``, used
throughout the package to convert scalar durations between units and to
compute conversion factors between frequencies/durations (central to
``HighFrequencyImputer`` / ``FrequencyConverter`` and to
``PublicationDelayTransformer``).

Calendar convention (documented in ``tsforecast/utils/abc/converter.py``,
``_CALENDAR_SUBPERIODS``): nested pairs among Y/Q/M/SM/W/D are EXACT
conventional counts (1 year = 12 months, 1 quarter = 3 months, 1 month = 30
days, 1 year = 365 days, ...), looked up before falling back to a
seconds-based ratio for any other pair. This makes ``Y -> M`` exactly ``12.0``
(not ``365 / 30 = 12.1667``): a divergence from ``notebooks/utils/
duration_converter.ipynb``, which predates this exact calendar table and is
outdated on this specific point (§2.3 of the campaign: the code prevails
over the notebooks when they disagree).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from tsforecast.utils.duration.converter import DurationConverter

_UNITS = ["ns", "us", "ms", "s", "min", "h", "D", "B", "W", "SM", "M", "Q", "Y"]


@pytest.fixture
def converter() -> DurationConverter:
    """A fresh ``DurationConverter`` instance."""
    return DurationConverter()


class TestConvert:
    """Contract of ``convert``: numeric value, unit -> numeric value, unit."""

    @pytest.mark.parametrize("unit", _UNITS)
    def test_identity_conversion(self, converter, unit):
        """Converting a unit to itself returns the value unchanged."""
        assert converter.convert(5, unit, unit) == 5

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            pytest.param(2.5, "h", "min", 150.0, id="hour-to-minute"),
            pytest.param(90, "min", "h", 1.5, id="minute-to-hour"),
            pytest.param(1.5, "D", "h", 36.0, id="day-to-hour"),
            pytest.param(1, "W", "D", 7.0, id="week-to-day"),
            pytest.param(1, "Y", "M", 12.0, id="year-to-month-exact-convention"),
            pytest.param(1, "Y", "D", 365.0, id="year-to-day-exact-convention"),
            pytest.param(1, "M", "D", 30.0, id="month-to-day-exact-convention"),
            pytest.param(1, "Q", "M", 3.0, id="quarter-to-month"),
        ],
    )
    def test_golden_conversions(self, converter, value, from_unit, to_unit, expected):
        """Table of golden conversions, computed by hand.

        Nested pairs use the exact factors of ``_CALENDAR_SUBPERIODS``.
        """
        assert converter.convert(value, from_unit, to_unit) == pytest.approx(expected)

    def test_mixed_code_and_literal_formats_agree(self, converter):
        """``convert`` accepts codes and literals alike.

        Both formats may be mixed between ``from_unit`` and ``to_unit``.
        """
        results = {
            "code-to-code": converter.convert(1, "D", "h"),
            "literal-to-literal": converter.convert(1, "day", "hour"),
            "code-to-literal": converter.convert(1, "D", "hour"),
            "literal-to-code": converter.convert(1, "day", "h"),
        }
        assert len(set(results.values())) == 1

    @pytest.mark.parametrize(
        "value, from_unit, to_unit",
        [
            pytest.param(3.7, "h", "min", id="fractional"),
            pytest.param(1000.0, "ns", "Y", id="extreme-magnitude-gap"),
            pytest.param(0.0, "D", "W", id="zero"),
        ],
    )
    def test_roundtrip_is_identity(self, converter, value, from_unit, to_unit):
        """``convert(convert(v, a, b), b, a) == v`` up to floating-point errors."""
        forward = converter.convert(value, from_unit, to_unit)
        backward = converter.convert(forward, to_unit, from_unit)
        assert backward == pytest.approx(value)

    def test_zero_value(self, converter):
        """A zero duration converts to zero."""
        assert converter.convert(0, "D", "h") == 0

    def test_negative_value_is_not_rejected(self, converter):
        """No sign validation is performed.

        A negative value (meaningless for a duration) is converted normally.
        """
        assert converter.convert(-5, "h", "min") == -300.0

    def test_extreme_magnitudes(self, converter):
        """Very large and very small values, without error nor loss of meaning."""
        assert converter.convert(1e12, "ns", "Y") == pytest.approx(1e12 * 1e-9 / 31536000)
        assert converter.convert(1e-6, "Y", "ns") == pytest.approx(1e-6 * 31536000 / 1e-9)

    @pytest.mark.parametrize(
        "rounding, expected_type, expected_value",
        [
            pytest.param(None, float, 37 / 24, id="no-rounding-returns-float"),
            pytest.param("floor", int, 1, id="floor-returns-int"),
            pytest.param("ceil", int, 2, id="ceil-returns-int"),
        ],
    )
    def test_rounding_modes(self, converter, rounding, expected_type, expected_value):
        """``rounding`` changes both the value and the return type.

        ``float`` without rounding, ``int`` with ``'floor'`` / ``'ceil'``
        (even when the exact result already is an integer).
        """
        result = converter.convert(37, "h", "D", rounding=rounding)
        assert type(result) is expected_type
        assert result == pytest.approx(expected_value)

    def test_rounding_follows_mathematical_convention_for_negative_values(self, converter):
        """``floor`` / ``ceil`` follow the standard mathematical convention.

        This includes negative values (floor towards -inf, ceil towards +inf).
        """
        assert converter.convert(-37, "h", "D", rounding="floor") == -2
        assert converter.convert(-37, "h", "D", rounding="ceil") == -1

    def test_unrecognized_rounding_value_raises(self, converter):
        """An unrecognized ``rounding`` value raises an explicit ``ValueError``.

        Hardening after ``ANO-UTILS-004`` (caveat raised in
        ``notebooks/utils/duration_converter.ipynb`` §3.4, originally ignored
        silently): a value other than ``'floor'``, ``'ceil'`` or ``None`` no
        longer silently returns the unrounded value.
        """
        with pytest.raises(ValueError, match="Unsupported rounding mode"):
            converter.convert(37, "h", "D", rounding="round")

    @pytest.mark.parametrize(
        "from_unit, to_unit",
        [
            pytest.param("foo", "D", id="unknown-from-unit"),
            pytest.param("D", "foo", id="unknown-to-unit"),
        ],
    )
    def test_unsupported_unit_raises(self, converter, from_unit, to_unit):
        """An unsupported unit (source or target) raises a ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            converter.convert(1, from_unit, to_unit)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("abc", id="string"),
            pytest.param(None, id="none"),
            pytest.param([1, 2, 3], id="list"),
        ],
    )
    def test_non_numeric_value_raises_type_error(self, converter, invalid_value):
        """A non-numeric value raises a native ``TypeError``.

        ``convert`` merely computes ``value * factor``: no dedicated message.
        """
        with pytest.raises(TypeError):
            converter.convert(invalid_value, "D", "h")


class TestMultipliedUnits:
    """A leading multiplier scales the unit it prefixes ('2D' is a unit of two days)."""

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            pytest.param(1, "2D", "h", 48.0, id="two-days-in-hours"),
            pytest.param(1, "h", "2h", 0.5, id="hour-in-two-hours"),
            pytest.param(3, "2D", "D", 6.0, id="value-times-multiplier"),
            pytest.param(1, "3M", "M", 3.0, id="three-months-in-months"),
            pytest.param(1, "Y", "3M", 4.0, id="year-in-quarters-of-months"),
            pytest.param(1, "2D", "2D", 1.0, id="same-multiplied-unit"),
            pytest.param(1, "2D", "4D", 0.5, id="same-code-two-multipliers"),
            pytest.param(1, "12MS", "Y", 1.0, id="twelve-months-in-a-year"),
        ],
    )
    def test_golden_conversions(self, converter, value, from_unit, to_unit, expected):
        """The multiplier of each unit enters the factor."""
        assert converter.convert(value, from_unit, to_unit) == pytest.approx(expected)

    def test_multiplier_one_is_transparent(self, converter):
        """'1D' is 'D'."""
        assert converter.get_conversion_factor("1D", "h") == converter.get_conversion_factor("D", "h")

    def test_factors_are_inverse_of_each_other(self, converter):
        """Swapping the units inverts the factor, multipliers included."""
        assert converter.get_conversion_factor("2D", "h") * converter.get_conversion_factor("h", "2D") == pytest.approx(1.0)

    def test_unsupported_unit_after_multiplier_raises(self, converter):
        """The multiplier does not make an unknown unit valid."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            converter.get_conversion_factor("2xyz", "h")


class TestGetConversionFactor:
    """Contract of ``get_conversion_factor``: (from_unit, to_unit) -> float."""

    @pytest.mark.parametrize(
        "from_unit, to_unit, expected",
        [
            # Paires emboîtées exactes (table _CALENDAR_SUBPERIODS)
            pytest.param("Y", "Q", 4.0, id="year-to-quarter"),
            pytest.param("Y", "M", 12.0, id="year-to-month"),
            pytest.param("Y", "SM", 24.0, id="year-to-semi-month"),
            pytest.param("Y", "W", 52.0, id="year-to-week"),
            pytest.param("Y", "D", 365.0, id="year-to-day"),
            pytest.param("Q", "M", 3.0, id="quarter-to-month"),
            pytest.param("Q", "SM", 6.0, id="quarter-to-semi-month"),
            pytest.param("Q", "W", 13.0, id="quarter-to-week"),
            pytest.param("Q", "D", 91.0, id="quarter-to-day"),
            pytest.param("M", "SM", 2.0, id="month-to-semi-month"),
            pytest.param("M", "D", 30.0, id="month-to-day"),
            pytest.param("W", "D", 7.0, id="week-to-day"),
            # Paires inversées : réciproque exacte de la table
            pytest.param("Q", "Y", 0.25, id="quarter-to-year-inverted"),
            pytest.param("D", "Y", 1 / 365, id="day-to-year-inverted"),
            pytest.param("D", "M", 1 / 30, id="day-to-month-inverted"),
            # Paires hors table : calcul via l'unité de référence (secondes)
            pytest.param("h", "min", 60.0, id="hour-to-minute"),
            pytest.param("D", "h", 24.0, id="day-to-hour"),
            pytest.param("Y", "h", 8760.0, id="year-to-hour-via-seconds"),
        ],
    )
    def test_golden_factors(self, converter, from_unit, to_unit, expected):
        """Table of golden factors.

        It crosses exact nested pairs, inverted pairs and pairs outside the
        table (fallback on seconds).
        """
        assert converter.get_conversion_factor(from_unit, to_unit) == pytest.approx(expected)

    def test_consistency_with_convert(self, converter):
        """``convert(v, a, b) == v * get_conversion_factor(a, b)``."""
        for value, from_unit, to_unit in [(3, "h", "min"), (2, "D", "h"), (1.5, "W", "D")]:
            factor = converter.get_conversion_factor(from_unit, to_unit)
            assert converter.convert(value, from_unit, to_unit) == pytest.approx(value * factor)

    def test_symmetry_for_all_unit_pairs(self, converter):
        """Property: ``factor(a, b) * factor(b, a) == 1`` for every pair of units.

        Whether the pair goes through the calendar table or through seconds.
        """
        for a, b in itertools.combinations(_UNITS, 2):
            factor_ab = converter.get_conversion_factor(a, b)
            factor_ba = converter.get_conversion_factor(b, a)
            assert np.isclose(factor_ab * factor_ba, 1.0)

    def test_business_day_treated_as_calendar_day(self, converter):
        """Documented limitation (notebook duration_converter §4.4): ``'B'`` is a calendar day.

        ``'B'`` (business day) is NOT in the nested calendar table and falls
        back on the seconds-based computation, identical to ``'D'``: no
        adjustment for non-business days (5/7).
        """
        assert converter.get_conversion_factor("B", "D") == 1.0
        assert converter.get_conversion_factor("B", "h") == converter.get_conversion_factor("D", "h")

    @pytest.mark.parametrize(
        "from_unit, to_unit",
        [
            pytest.param("foo", "D", id="unknown-from-unit"),
            pytest.param("D", "foo", id="unknown-to-unit"),
        ],
    )
    def test_unsupported_unit_raises(self, converter, from_unit, to_unit):
        """An unsupported unit raises a ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            converter.get_conversion_factor(from_unit, to_unit)
