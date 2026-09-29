"""Tests for ``tsforecast.utils.duration.utils``.

Covers the module-level convenience functions that delegate to a shared
``DurationNormalizer``/``DurationConverter`` instance (or a fresh one, for
``get_duration_conversion_factor``/``convert_duration``): ``normalize_duration``,
``to_literal``, ``to_code``, ``validate_duration``, ``get_duration_conversion_factor``,
``convert_duration`` and ``get_duration_order``. Equivalence with the class
methods is the central property under test: these functions add no logic of
their own beyond instantiation and delegation.
"""
from __future__ import annotations

import pytest

from tsforecast.utils.duration.converter import DurationConverter
from tsforecast.utils.duration.normalizer import DurationNormalizer
from tsforecast.utils.duration.utils import (
    convert_duration,
    get_duration_conversion_factor,
    get_duration_order,
    normalize_duration,
    to_code,
    to_literal,
    validate_duration,
)

_CODES = ["ns", "us", "ms", "s", "min", "h", "D", "B", "W", "SM", "M", "Q", "Y"]
_LITERALS = [
    "nanosecond", "microsecond", "millisecond", "second", "minute", "hour",
    "day", "business_day", "week", "semi_month", "month", "quarter", "year",
]


class TestNormalizeDuration:
    """``normalize_duration`` delegates to ``DurationNormalizer.normalize``."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_matches_normalizer_instance(self, value):
        """Same result as a dedicated ``DurationNormalizer`` instance."""
        assert normalize_duration(value) == DurationNormalizer().normalize(value)

    def test_unsupported_value_raises(self):
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalize_duration("xyz")


class TestToCodeAndToLiteralFunctions:
    """``to_code``/``to_literal`` module functions delegate to the class methods."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_to_code_matches_normalizer_instance(self, value):
        assert to_code(value) == DurationNormalizer().to_code(value)

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_to_literal_matches_normalizer_instance(self, value):
        assert to_literal(value) == DurationNormalizer().to_literal(value)

    def test_to_literal_unsupported_value_raises(self):
        with pytest.raises(ValueError, match="Unsupported duration"):
            to_literal("xyz")


class TestValidateDuration:
    """``validate_duration`` delegates to ``DurationNormalizer.validate``."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param("day", True, id="valid-literal"),
            pytest.param("h", True, id="valid-code"),
            pytest.param("invalid_dur", False, id="invalid-string"),
            pytest.param(None, False, id="none"),
        ],
    )
    def test_matches_normalizer_instance(self, value, expected):
        assert validate_duration(value) is expected is DurationNormalizer().validate(value)


class TestGetDurationConversionFactor:
    """``get_duration_conversion_factor`` delegates to ``DurationConverter``."""

    @pytest.mark.parametrize(
        "from_duration, to_duration, expected",
        [
            pytest.param("hour", "minute", 60.0, id="hour-to-minute"),
            pytest.param("day", "hour", 24.0, id="day-to-hour"),
            pytest.param("Y", "M", 12.0, id="year-to-month-exact-convention"),
        ],
    )
    def test_golden_factors(self, from_duration, to_duration, expected):
        assert get_duration_conversion_factor(from_duration, to_duration) == pytest.approx(expected)

    def test_matches_converter_instance(self):
        assert get_duration_conversion_factor("Q", "M") == DurationConverter().get_conversion_factor("Q", "M")

    def test_unsupported_unit_raises(self):
        with pytest.raises(ValueError, match="Unsupported duration"):
            get_duration_conversion_factor("foo", "D")


class TestConvertDuration:
    """``convert_duration`` delegates to ``DurationConverter.convert``."""

    @pytest.mark.parametrize(
        "value, from_duration, to_duration, expected",
        [
            pytest.param(2.5, "hour", "minute", 150.0, id="hour-to-minute"),
            pytest.param(90, "minute", "hour", 1.5, id="minute-to-hour"),
            pytest.param(1.5, "day", "hour", 36.0, id="day-to-hour"),
        ],
    )
    def test_golden_conversions(self, value, from_duration, to_duration, expected):
        assert convert_duration(value, from_duration, to_duration) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "rounding, expected",
        [
            pytest.param(None, 1.5, id="no-rounding"),
            pytest.param("floor", 1, id="floor"),
            pytest.param("ceil", 2, id="ceil"),
        ],
    )
    def test_rounding_is_forwarded(self, rounding, expected):
        """The ``rounding`` parameter is forwarded to the underlying converter."""
        assert convert_duration(90, "minute", "hour", rounding=rounding) == pytest.approx(expected)

    def test_matches_converter_instance(self):
        assert convert_duration(90, "minute", "hour", rounding="ceil") == (
            DurationConverter().convert(90, "minute", "hour", rounding="ceil")
        )

    def test_unsupported_unit_raises(self):
        with pytest.raises(ValueError, match="Unsupported duration"):
            convert_duration(1, "foo", "D")


class TestGetDurationOrder:
    """``get_duration_order`` reads ``DurationNormalizer._duration_order`` directly.

    Fragile coupling point raised by the ``duration_normalizer.ipynb``
    notebook: unlike the other functions of this module,
    ``get_duration_order`` goes through no public method of
    ``DurationNormalizer`` to read the order of a code.
    """

    @pytest.mark.parametrize(
        "duration, expected_order",
        [
            pytest.param("day", 7.0, id="day"),
            pytest.param("month", 9.0, id="month"),
            pytest.param("hour", 6.0, id="hour"),
            pytest.param("business_day", 7.5, id="business-day-non-integer-order"),
            pytest.param("semi_month", 8.5, id="semi-month-non-integer-order"),
        ],
    )
    def test_golden_orders(self, duration, expected_order):
        assert get_duration_order(duration) == expected_order

    def test_quarter_order_greater_than_month_order(self):
        assert get_duration_order("quarter") > get_duration_order("month")

    def test_consistent_with_is_longer_duration_for_every_pair(self):
        """Property: ``get_duration_order(a) > get_duration_order(b)`` iff ``is_longer_duration(a, b)``.

        Checked for every pair of codes against
        ``DurationNormalizer().is_longer_duration``.
        """
        normalizer = DurationNormalizer()
        for a in _CODES:
            for b in _CODES:
                assert (get_duration_order(a) > get_duration_order(b)) == normalizer.is_longer_duration(a, b)

    def test_unknown_duration_raises_via_normalize_duration(self):
        """An unknown duration raises through ``normalize_duration``.

        ``get_duration_order`` first calls ``normalize_duration``, which
        raises a ``ValueError``: the internal ``.get(code, 0)`` fallback is
        never reached for an invalid input (dead code with the current
        mappings).
        """
        with pytest.raises(ValueError, match="Unsupported duration"):
            get_duration_order("xyz")
