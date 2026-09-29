"""Tests for ``tsforecast.utils.duration.normalizer.DurationNormalizer``.

Covers the public methods of ``DurationNormalizer``: ``normalize`` (and its
alias ``to_code``), ``to_literal``, ``validate``, ``is_longer_duration`` and
``are_compatible_durations``. The class centralizes conversions between
duration codes (``'D'``, ``'M'``, ``'Q'``...) and user-friendly literal names
(``'day'``, ``'month'``, ``'quarter'``...), with a fallback onto
``parse_frequency`` for full pandas frequency strings (``'MS'``, ``'QE-DEC'``).
"""
from __future__ import annotations

import itertools

import pytest

from tsforecast.utils.duration.normalizer import DurationNormalizer

# Codes déclarés par DurationType, dans l'ordre croissant de durée (cf.
# _duration_order de DurationNormalizer : ns < us < ms < s < min < h < D < B
# < W < SM < M < Q < Y).
_CODES = ["ns", "us", "ms", "s", "min", "h", "D", "B", "W", "SM", "M", "Q", "Y"]
_LITERALS = [
    "nanosecond", "microsecond", "millisecond", "second", "minute", "hour",
    "day", "business_day", "week", "semi_month", "month", "quarter", "year",
]


@pytest.fixture
def normalizer() -> DurationNormalizer:
    """A fresh ``DurationNormalizer`` instance."""
    return DurationNormalizer()


class TestNormalize:
    """Contract of ``normalize``: any supported representation -> code."""

    @pytest.mark.parametrize("code", _CODES)
    def test_code_is_returned_unchanged(self, normalizer, code):
        """An already normalized code is returned as is."""
        assert normalizer.normalize(code) == code

    @pytest.mark.parametrize(
        "literal, expected_code",
        list(zip(_LITERALS, _CODES)),
        ids=_LITERALS,
    )
    def test_literal_resolves_to_code(self, normalizer, literal, expected_code):
        """Each literal name resolves to its code."""
        assert normalizer.normalize(literal) == expected_code

    @pytest.mark.parametrize(
        "frequency_str, expected_code",
        [
            pytest.param("MS", "M", id="monthly-start"),
            pytest.param("ME", "M", id="monthly-end"),
            pytest.param("QE-DEC", "Q", id="quarterly-end-dec-anchor"),
            pytest.param("QS-JAN", "Q", id="quarterly-start-jan-anchor"),
            pytest.param("YE-DEC", "Y", id="yearly-end-dec-anchor"),
            pytest.param("W-MON", "W", id="weekly-monday-anchor"),
        ],
    )
    def test_falls_back_to_parse_frequency_for_pandas_strings(
        self, normalizer, frequency_str, expected_code
    ):
        """Fallback on ``parse_frequency`` for full pandas frequency strings.

        Position and anchor suffix are ignored, only the base frequency is
        kept.
        """
        assert normalizer.normalize(frequency_str) == expected_code

    def test_uppercase_s_is_not_recognized_as_seconds(self, normalizer):
        """Pinned pitfall (notebook duration_normalizer): the seconds code is lowercase ``'s'``.

        Uppercase ``'S'`` is neither a known code nor a literal, and the
        ``parse_frequency('S')`` fallback extracts a base identical to the
        input: the infinite-recursion guard stops there, so normalization
        fails instead of falling back on seconds.
        """
        assert normalizer.normalize("s") == "s"
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.normalize("S")

    @pytest.mark.parametrize(
        "invalid_code",
        [
            pytest.param("d", id="lowercase-d"),
            pytest.param("HOUR", id="uppercase-hour"),
            pytest.param("Hour", id="titlecase-hour"),
            pytest.param("Min", id="titlecase-min"),
        ],
    )
    def test_case_sensitive_rejection(self, normalizer, invalid_code):
        """No case normalization is applied.

        A wrongly capitalized variant of an otherwise valid code or literal is
        rejected.
        """
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.normalize(invalid_code)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(5, id="int"),
            pytest.param(3.14, id="float"),
            pytest.param(None, id="none"),
            pytest.param(["D"], id="list"),
        ],
    )
    def test_non_string_input_raises(self, normalizer, invalid_value):
        """A non-``str`` input is explicitly rejected.

        No ``TypeError`` is left to propagate naturally, unlike
        ``DurationConverter``.
        """
        with pytest.raises(ValueError, match="must be a string"):
            normalizer.normalize(invalid_value)

    @pytest.mark.parametrize(
        "invalid_value",
        [pytest.param("xyz", id="unknown-code"), pytest.param("", id="empty-string")],
    )
    def test_unsupported_string_raises(self, normalizer, invalid_value):
        """An unrecognized string, even after the ``parse_frequency`` fallback, raises an explicit error."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.normalize(invalid_value)

    @pytest.mark.parametrize("value, expected", [("2D", "D"), ("3MS", "M"), ("12ME", "M"), ("15min", "min")])
    def test_multiplied_duration_is_normalized_to_its_code(self, normalizer, value, expected):
        """A leading multiplier is accepted and left out of the code."""
        assert normalizer.normalize(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [("2D", ("D", 2)), ("3MS", ("M", 3)), ("day", ("D", 1)), ("business_day", ("B", 1)), ("min", ("min", 1))],
    )
    def test_normalize_with_multiplier(self, normalizer, value, expected):
        """The code and the multiplier are returned together."""
        assert normalizer.normalize_with_multiplier(value) == expected

    @pytest.mark.parametrize("value", ["2xyz", ""])
    def test_multiplied_unsupported_duration_raises(self, normalizer, value):
        """The multiplier does not make an unknown unit valid."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.normalize(value)

    @pytest.mark.parametrize(
        "dur1, dur2, expected",
        [
            pytest.param("2D", "D", True, id="same-code-higher-multiplier"),
            pytest.param("D", "2D", False, id="same-code-lower-multiplier"),
            pytest.param("2D", "W", False, id="two-days-vs-week"),
            pytest.param("8D", "W", True, id="eight-days-vs-week"),
            pytest.param("120min", "h", True, id="two-hours-in-minutes-vs-hour"),
            pytest.param("month", "day", True, id="no-multiplier"),
        ],
    )
    def test_is_longer_duration_with_multiplier(self, normalizer, dur1, dur2, expected):
        """A multiplier lengthens the duration."""
        assert normalizer.is_longer_duration(dur1, dur2) is expected


class TestToCode:
    """``to_code`` is a strict alias of ``normalize``."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_matches_normalize_on_valid_values(self, normalizer, value):
        """Same result as ``normalize`` for every valid input."""
        assert normalizer.to_code(value) == normalizer.normalize(value)

    def test_matches_normalize_error_on_invalid_value(self, normalizer):
        """Same error as ``normalize`` for an invalid input."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.to_code("xyz")


class TestToLiteral:
    """Contract of ``to_literal``: any supported representation -> literal."""

    @pytest.mark.parametrize(
        "code, expected_literal", list(zip(_CODES, _LITERALS)), ids=_CODES
    )
    def test_code_resolves_to_literal(self, normalizer, code, expected_literal):
        """A code resolves to its literal name."""
        assert normalizer.to_literal(code) == expected_literal

    @pytest.mark.parametrize("literal", _LITERALS)
    def test_literal_is_idempotent(self, normalizer, literal):
        """An already literal value resolves to itself."""
        assert normalizer.to_literal(literal) == literal

    @pytest.mark.parametrize("code", _CODES)
    def test_roundtrip_to_code_to_literal_is_identity(self, normalizer, code):
        """``to_code(to_literal(code)) == code``: code <-> literal bijection."""
        assert normalizer.to_code(normalizer.to_literal(code)) == code

    def test_invalid_value_raises(self, normalizer):
        """The ``normalize`` error is propagated unchanged."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.to_literal("xyz")


class TestValidate:
    """``validate`` never raises; it wraps ``normalize`` in a try/except."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS + ["MS", "QE-DEC", "W-MON"])
    def test_true_for_supported_values(self, normalizer, value):
        """Every value accepted by ``normalize`` is valid."""
        assert normalizer.validate(value) is True

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("xyz", id="unknown-string"),
            pytest.param("d", id="wrong-case"),
            pytest.param(None, id="none"),
            pytest.param(5, id="int"),
            pytest.param(3.14, id="float"),
            pytest.param(["D"], id="list"),
            pytest.param({"a": 1}, id="dict"),
        ],
    )
    def test_false_without_raising_for_unsupported_values(self, normalizer, invalid_value):
        """No exception propagates, even for obviously invalid types.

        ``validate`` always returns a boolean.
        """
        assert normalizer.validate(invalid_value) is False


class TestIsLongerDuration:
    """Strict ordering of durations, from ``_duration_order``."""

    @pytest.mark.parametrize(
        "shorter, longer",
        list(zip(_CODES[:-1], _CODES[1:])),
        ids=[f"{a}-lt-{b}" for a, b in zip(_CODES[:-1], _CODES[1:])],
    )
    def test_consecutive_codes_are_strictly_ordered(self, normalizer, shorter, longer):
        """The declared order (ns < us < ... < B < W < SM < M < Q < Y) holds between consecutive codes.

        Checked in both directions.
        """
        assert normalizer.is_longer_duration(longer, shorter) is True
        assert normalizer.is_longer_duration(shorter, longer) is False

    def test_mixed_code_and_literal_formats(self, normalizer):
        """Both arguments may mix code and literal."""
        assert normalizer.is_longer_duration("month", "day") is True
        assert normalizer.is_longer_duration("M", "D") is True
        assert normalizer.is_longer_duration("month", "D") is True
        assert normalizer.is_longer_duration("day", "month") is False

    @pytest.mark.parametrize("code", _CODES)
    def test_equality_is_always_false(self, normalizer, code):
        """Strict comparison (``>``): ``is_longer_duration(x, x)`` is always ``False``.

        Never ``>=``.
        """
        assert normalizer.is_longer_duration(code, code) is False

    def test_total_order_is_transitive(self, normalizer):
        """Property: the relation is a total order on the codes.

        Antisymmetric and transitive, consistent with the order declared in
        ``_CODES``.
        """
        for a, b in itertools.combinations(_CODES, 2):
            index_a, index_b = _CODES.index(a), _CODES.index(b)
            expected = index_a > index_b
            assert normalizer.is_longer_duration(a, b) is expected
            # Antisymétrie : jamais vrai dans les deux sens à la fois
            assert not (
                normalizer.is_longer_duration(a, b)
                and normalizer.is_longer_duration(b, a)
            )

    def test_invalid_duration_raises_before_fallback_to_zero(self, normalizer):
        """An invalid duration raises before the internal fallback to zero.

        ``is_longer_duration`` calls ``to_code`` on both arguments, which
        raises a ``ValueError`` before the internal fallback
        ``_duration_order.get(code, 0)`` is reached: an invalid duration is
        never silently treated as "the shortest".
        """
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.is_longer_duration("day", "xyz")
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.is_longer_duration("xyz", "day")


class TestAreCompatibleDurations:
    """``are_compatible_durations`` is strictly ``validate(a) and validate(b)``.

    The name suggests a joint compatibility check (does converting ``dur1``
    to ``dur2`` make sense?), but the implementation only normalizes each
    duration independently, without ever comparing the two results.
    """

    @pytest.mark.parametrize(
        "dur1, dur2",
        [
            pytest.param("day", "month", id="both-valid"),
            pytest.param("ns", "Y", id="both-valid-extreme-orders-of-magnitude"),
            pytest.param("business_day", "month", id="both-valid-mixed-format"),
            pytest.param("day", "xyz", id="second-invalid"),
            pytest.param("xyz", "day", id="first-invalid"),
            pytest.param("xyz", "abc", id="both-invalid"),
        ],
    )
    def test_matches_validate_conjunction(self, normalizer, dur1, dur2):
        """Strict equivalence with ``validate(dur1) and validate(dur2)``.

        No notion of a reasonable conversion ratio between both durations is
        checked (e.g. ``'ns'`` / ``'Y'`` are deemed "compatible").
        """
        expected = normalizer.validate(dur1) and normalizer.validate(dur2)
        assert normalizer.are_compatible_durations(dur1, dur2) == expected
