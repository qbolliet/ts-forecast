"""Tests for ``tsforecast.utils.parse.utils``.

This module covers the two only primitives that manipulate pandas frequency
strings as plain text: ``parse_frequency`` (splits a frequency string into its
base frequency, position and suffix components) and ``build_frequency_string``
(reassembles those components into a pandas-compatible frequency string).
Coverage includes the parse/build round trip for every supported frequency
family (calendar and sub-daily), anchored offsets, deliberate rejections
(pandas ``'A'``/``'AS'``/``'AE'`` aliases, a null or lone multiplier), leading
multipliers (``'2MS'``) and casing sensitivity.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from tsforecast.utils.parse.utils import ParsedFrequency, build_frequency_string, parse_frequency


class TestParseFrequency:
    """Contract of ``parse_frequency``: string -> (freq, position, suffix, multiplier)."""

    @pytest.mark.parametrize(
        "frequency_str, expected",
        [
            pytest.param("D", ("D", None, None), id="daily"),
            pytest.param("W", ("W", None, None), id="weekly-no-anchor"),
            pytest.param("MS", ("M", "S", None), id="monthly-start"),
            pytest.param("ME", ("M", "E", None), id="monthly-end"),
            pytest.param("QS", ("Q", "S", None), id="quarterly-start"),
            pytest.param("QE", ("Q", "E", None), id="quarterly-end"),
            pytest.param("YS", ("Y", "S", None), id="yearly-start"),
            pytest.param("YE", ("Y", "E", None), id="yearly-end"),
            pytest.param("B", ("B", None, None), id="business-daily"),
            # Sous-journalières : aucune ne porte de position S/E, l'indicateur
            # est conservé tel quel, en minuscules
            pytest.param("h", ("h", None, None), id="hourly"),
            pytest.param("min", ("min", None, None), id="minute"),
            pytest.param("s", ("s", None, None), id="second"),
            pytest.param("ms", ("ms", None, None), id="millisecond"),
            pytest.param("us", ("us", None, None), id="microsecond"),
            pytest.param("ns", ("ns", None, None), id="nanosecond"),
        ],
    )
    def test_splits_frequency_position_and_suffix(self, frequency_str, expected):
        """Base frequency, position and suffix are extracted correctly (multiplier 1)."""
        assert parse_frequency(frequency_str) == (*expected, 1)

    @pytest.mark.parametrize(
        "frequency_str, expected",
        [
            pytest.param("QE-DEC", ("Q", "E", "DEC"), id="quarterly-end-dec-anchor"),
            pytest.param("QS-JAN", ("Q", "S", "JAN"), id="quarterly-start-jan-anchor"),
            # Les ancres hebdomadaires (jour de la semaine) n'ont pas de notion
            # de position S/E en pandas : seul le suffixe porte l'information
            pytest.param("W-MON", ("W", None, "MON"), id="weekly-monday-anchor"),
            pytest.param("W-SUN", ("W", None, "SUN"), id="weekly-sunday-anchor"),
        ],
    )
    def test_splits_anchored_frequency(self, frequency_str, expected):
        """Anchor suffixes (quarter month, week day) are extracted correctly."""
        assert parse_frequency(frequency_str) == (*expected, 1)

    def test_none_input_raises_could_not_detect(self):
        """A ``None`` input (no frequency detected upstream) raises explicitly."""
        with pytest.raises(ValueError, match="Could not detect"):
            parse_frequency(None)

    def test_empty_string_raises_unable_to_parse(self):
        """An empty string does not match the expected grammar."""
        with pytest.raises(ValueError, match="Unable to parse"):
            parse_frequency("")

    @pytest.mark.parametrize(
        "frequency_str, expected",
        [
            pytest.param("2MS", ("M", "S", None, 2), id="bimonthly-start"),
            pytest.param("12ME", ("M", "E", None, 12), id="twelve-months-end"),
            pytest.param("3QS-FEB", ("Q", "S", "FEB", 3), id="multiplier-and-anchor"),
            pytest.param("2W-MON", ("W", None, "MON", 2), id="biweekly-monday"),
            pytest.param("2D", ("D", None, None, 2), id="two-days"),
            pytest.param("15min", ("min", None, None, 15), id="fifteen-minutes"),
            pytest.param("2SMS", ("SM", "S", None, 2), id="multiplied-semi-monthly"),
            # Un 1 explicite équivaut à l'absence de multiplicateur
            pytest.param("1MS", ("M", "S", None, 1), id="explicit-one"),
        ],
    )
    def test_leading_multiplier_is_extracted(self, frequency_str, expected):
        """A leading integer is the multiplier, kept out of the base frequency code."""
        assert parse_frequency(frequency_str) == expected

    def test_result_is_a_named_tuple(self):
        """Components are reachable by name and keep the positional order."""
        parsed = parse_frequency("2QS-FEB")
        assert isinstance(parsed, ParsedFrequency)
        assert (parsed.freq, parsed.position, parsed.suffix, parsed.multiplier) == ("Q", "S", "FEB", 2)

    def test_multiplier_defaults_to_one(self):
        """Building a ParsedFrequency without multiplier gives 1."""
        assert ParsedFrequency("M", "S", None).multiplier == 1

    @pytest.mark.parametrize("frequency_str", ["0MS", "00D"])
    def test_null_multiplier_raises(self, frequency_str):
        """A null multiplier is meaningless for pandas and rejected."""
        with pytest.raises(ValueError, match="positive integer"):
            parse_frequency(frequency_str)

    @pytest.mark.parametrize("frequency_str", ["2", "12", "2-DEC"])
    def test_multiplier_without_frequency_raises(self, frequency_str):
        """A multiplier alone does not describe a frequency."""
        with pytest.raises(ValueError, match="Unable to parse"):
            parse_frequency(frequency_str)

    @pytest.mark.parametrize(
        "frequency_str, expected",
        [
            # 'S' et 'E' majuscules seuls sont interprétés comme fréquence de
            # base (pas de position) : la fréquence doit contenir au moins un
            # caractère avant l'indicateur de position pour que ce dernier soit
            # reconnu comme tel
            pytest.param("S", ("S", None, None), id="bare-s-is-a-frequency"),
            pytest.param("E", ("E", None, None), id="bare-e-is-a-frequency"),
        ],
    )
    def test_bare_position_letters_are_not_positions(self, frequency_str, expected):
        """A lone 'S'/'E' is parsed as a base frequency code, not a position."""
        assert parse_frequency(frequency_str) == (*expected, 1)

    def test_case_sensitivity_lowercase_ms_is_not_month_start(self):
        """Case matters: 'ms' (milliseconds) != 'MS' (month start)."""
        assert parse_frequency("ms") == ("ms", None, None, 1)
        assert parse_frequency("MS") == ("M", "S", None, 1)


class TestBuildFrequencyString:
    """Contract of ``build_frequency_string``: (freq, position, suffix, multiplier) -> string."""

    @pytest.mark.parametrize(
        "frequency, position, suffix, expected",
        [
            pytest.param("D", None, None, "D", id="daily-no-position"),
            pytest.param("W", None, None, "W", id="weekly-no-position"),
            pytest.param("M", "S", None, "MS", id="monthly-start"),
            pytest.param("M", "E", None, "ME", id="monthly-end"),
            pytest.param("Q", "S", None, "QS", id="quarterly-start"),
            pytest.param("Q", "E", None, "QE", id="quarterly-end"),
            pytest.param("Y", "S", None, "YS", id="yearly-start"),
            pytest.param("Y", "E", None, "YE", id="yearly-end"),
            # Jour ouvré et semaine : pas de variante S/E en pandas ('BE', 'WS'
            # n'existent pas), la position est ignorée (ANO-UTILS-009)
            pytest.param("B", "E", None, "B", id="business-daily-position-ignored"),
            pytest.param("W", "S", None, "W", id="weekly-position-ignored"),
            pytest.param("W", "E", "MON", "W-MON", id="weekly-anchor-position-ignored"),
            # Semi-mensuel : grilles pandas SMS / SME
            pytest.param("SM", "S", None, "SMS", id="semi-monthly-start"),
            pytest.param("SM", "E", None, "SME", id="semi-monthly-end"),
            pytest.param("Q", "E", "DEC", "QE-DEC", id="quarterly-end-dec-anchor"),
            pytest.param("Q", "S", "JAN", "QS-JAN", id="quarterly-start-jan-anchor"),
            # Fréquences sous-journalières : non "position-aware", la position
            # demandée est silencieusement ignorée (§ liste _POSITION_AWARE_FREQUENCIES)
            pytest.param("h", "S", None, "h", id="hourly-position-ignored"),
            pytest.param("min", "E", None, "min", id="minute-position-ignored"),
            pytest.param("s", None, None, "s", id="second-no-position"),
            pytest.param("ms", None, None, "ms", id="millisecond-no-position"),
            pytest.param("us", None, None, "us", id="microsecond-no-position"),
            pytest.param("ns", None, None, "ns", id="nanosecond-no-position"),
        ],
    )
    def test_builds_expected_string(self, frequency, position, suffix, expected):
        """Base frequency, position and suffix are reassembled correctly."""
        assert build_frequency_string(frequency, position=position, suffix=suffix) == expected

    @pytest.mark.parametrize(
        "frequency, position, suffix, multiplier, expected",
        [
            pytest.param("M", "S", None, 2, "2MS", id="bimonthly-start"),
            pytest.param("Q", "S", "FEB", 3, "3QS-FEB", id="multiplier-and-anchor"),
            pytest.param("W", None, "MON", 2, "2W-MON", id="biweekly-monday"),
            pytest.param("min", None, None, 15, "15min", id="fifteen-minutes"),
            # Le multiplicateur 1 est omis
            pytest.param("M", "E", None, 1, "ME", id="multiplier-one-omitted"),
        ],
    )
    def test_multiplier_is_put_in_front(self, frequency, position, suffix, multiplier, expected):
        """A multiplier above 1 prefixes the string; 1 is left out."""
        assert build_frequency_string(frequency, position, suffix, multiplier) == expected

    @pytest.mark.parametrize("multiplier", [0, -2, 1.5, "2", True, None])
    def test_invalid_multiplier_raises(self, multiplier):
        """The multiplier must be a positive integer (booleans excluded)."""
        with pytest.raises(ValueError, match="multiplier must be"):
            build_frequency_string("M", multiplier=multiplier)

    def test_suffix_kept_without_position(self):
        """The suffix is always appended when given, even without a position.

        Required for anchors without an S/E position notion (e.g. a weekday,
        ``'W-MON'``) - see ``ANO-UTILS-001``, fixed.
        """
        assert build_frequency_string("Q", position=None, suffix="DEC") == "Q-DEC"
        assert build_frequency_string("W", position=None, suffix="MON") == "W-MON"

    def test_suffix_kept_even_when_position_is_silently_ignored(self):
        """Pinned current behaviour: for a frequency without position, the suffix is still appended.

        The requested position is ignored, but the suffix is appended anyway
        (code path independent of the ``_POSITION_AWARE_FREQUENCIES`` check).
        """
        assert build_frequency_string("h", position="S", suffix="FOO") == "h-FOO"

    @pytest.mark.parametrize(
        "invalid_position",
        [
            pytest.param("start", id="literal-start-refused"),
            pytest.param("end", id="literal-end-refused"),
            pytest.param("s", id="lowercase-s-refused"),
            pytest.param("AS", id="pandas-alias-as-not-a-position"),
            pytest.param("X", id="arbitrary-invalid-position"),
        ],
    )
    def test_invalid_position_raises(self, invalid_position):
        """Only 'S', 'E' and None are valid positions."""
        with pytest.raises(ValueError, match="position must be"):
            build_frequency_string("D", position=invalid_position)

    @pytest.mark.parametrize(
        "frequency",
        [
            pytest.param("A", id="deprecated-annual-alias"),
            pytest.param("AS", id="deprecated-annual-start-alias"),
            pytest.param("AE", id="deprecated-annual-end-alias"),
        ],
    )
    def test_deprecated_annual_aliases_rejected(self, frequency):
        """Explicit and deliberate rejection of the pandas aliases 'A' / 'AS' / 'AE'.

        Consolidation of ``parse_frequency`` / ``build_frequency_string`` as
        the only frequency primitives, accepted by the author: the base
        frequency goes through ``normalize_frequency``, which no longer
        recognizes them.
        """
        with pytest.raises(ValueError, match="Unsupported frequency"):
            build_frequency_string(frequency)

    def test_unsupported_base_frequency_raises(self):
        """An unknown base frequency is rejected by the normalization."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            build_frequency_string("not_a_frequency")

    def test_case_insensitive_base_frequency_normalization(self):
        """The base frequency goes through ``normalize_frequency``.

        Literal names ('monthly') are accepted just like codes.
        """
        assert build_frequency_string("monthly", position="S") == "MS"


class TestParseBuildRoundTrip:
    """Round trip ``parse_frequency`` -> ``build_frequency_string``.

    Rebuilding from the parsed components reproduces the original string
    exactly for every supported frequency string, including anchors without
    an S/E position such as a weekday anchor (``'W-MON'``, fixed by
    ``ANO-UTILS-001``).
    """

    @pytest.mark.parametrize(
        "frequency_str",
        [
            "D", "W", "B",
            "MS", "ME", "QS", "QE", "YS", "YE",
            "h", "min", "s", "ms", "us", "ns",
            "QE-DEC", "QS-JAN", "W-MON", "W-SUN", "SMS", "SME",
            "2MS", "12ME", "3QS-FEB", "2W-MON", "2D", "15min", "2SMS",
        ],
    )
    def test_roundtrip_is_identity(self, frequency_str):
        """Parse then build reproduces the original string exactly."""
        assert build_frequency_string(*parse_frequency(frequency_str)) == frequency_str


class TestBuiltStringIsValidPandas:
    """``build_frequency_string`` only produces aliases understood by pandas."""

    @pytest.mark.parametrize("position", ["S", "E", None])
    @pytest.mark.parametrize("frequency", ["D", "B", "W", "SM", "M", "Q", "Y", "h", "min", "s"])
    def test_every_base_and_position_gives_a_valid_offset(self, frequency, position):
        """Any supported base frequency combined with any position is a valid pandas offset.

        Before the fix (ANO-UTILS-009), ``'W'`` and ``'B'`` received a position
        suffix (``'WS'``, ``'BE'``) unknown to pandas.
        """
        assert to_offset(build_frequency_string(frequency, position=position)) is not None


class TestDefaultPosition:
    """``default_position`` turns a bare base code into a pandas alias (period end by default).

    A bare ``'M'`` / ``'Q'`` / ``'Y'`` / ``'SM'`` is a base code, not a pandas alias any more
    (deprecated in pandas 2.2, removed in pandas 3). Callers that hand the string to pandas
    pass ``default_position='E'``; the others (durations, comparisons) keep the bare code.
    """

    @pytest.mark.parametrize("frequency", ["M", "Q", "Y", "SM"])
    def test_bare_code_stays_bare_without_default_position(self, frequency):
        """Without ``default_position`` the string is the faithful assembly of its components."""
        assert build_frequency_string(frequency) == frequency

    @pytest.mark.parametrize(
        "frequency, expected",
        [("M", "ME"), ("Q", "QE"), ("Y", "YE"), ("SM", "SME")],
    )
    def test_end_default(self, frequency, expected):
        """Position-aware frequencies get the period-end variant."""
        assert build_frequency_string(frequency, default_position="E") == expected

    @pytest.mark.parametrize(
        "frequency, expected",
        [("M", "MS"), ("Q", "QS"), ("Y", "YS"), ("SM", "SMS")],
    )
    def test_start_default(self, frequency, expected):
        """The default position can be the period start."""
        assert build_frequency_string(frequency, default_position="S") == expected

    @pytest.mark.parametrize("position", ["S", "E"])
    @pytest.mark.parametrize("default_position", ["S", "E", None])
    def test_explicit_position_wins(self, position, default_position):
        """``default_position`` never overrides an explicit position."""
        assert build_frequency_string("M", position, default_position=default_position) == f"M{position}"

    @pytest.mark.parametrize("frequency", ["D", "B", "W", "h", "min", "s", "ms", "us", "ns"])
    def test_ignored_without_start_end_variant(self, frequency):
        """Frequencies pandas has no start / end variant for stay unchanged."""
        assert build_frequency_string(frequency, default_position="E") == frequency

    @pytest.mark.parametrize(
        "frequency, suffix, multiplier, expected",
        [
            ("Q", "DEC", 1, "QE-DEC"),
            ("Q", "FEB", 3, "3QE-FEB"),
            ("M", None, 2, "2ME"),
            ("W", "MON", 2, "2W-MON"),
        ],
    )
    def test_suffix_and_multiplier_are_kept(self, frequency, suffix, multiplier, expected):
        """The anchor and the multiplier go through unchanged."""
        assert build_frequency_string(frequency, suffix=suffix, multiplier=multiplier, default_position="E") == expected

    @pytest.mark.parametrize("default_position", ["X", "start", "s", ""])
    def test_invalid_default_position_raises(self, default_position):
        """Only ``'S'``, ``'E'`` and None are accepted."""
        with pytest.raises(ValueError, match="default_position"):
            build_frequency_string("M", default_position=default_position)

    @pytest.mark.parametrize("frequency", ["D", "B", "W", "SM", "M", "Q", "Y", "h", "min", "s"])
    @pytest.mark.parametrize("default_position", ["S", "E"])
    def test_result_is_a_pandas_alias_without_warning(self, frequency, default_position):
        """Property: with a default position, pandas reads the string and does not deprecate it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert to_offset(build_frequency_string(frequency, default_position=default_position)) is not None

    @pytest.mark.parametrize("frequency", ["M", "Q", "Y", "SM"])
    def test_bare_code_is_deprecated_by_pandas(self, frequency):
        """Why the parameter exists: the bare code makes pandas warn (and fails in pandas 3)."""
        with pytest.warns(FutureWarning, match="deprecated"):
            to_offset(build_frequency_string(frequency))

    @pytest.mark.parametrize("frequency", ["M", "Q", "Y", "SM"])
    def test_end_alias_generates_the_dates_of_the_former_bare_alias(self, frequency):
        """Property: the end alias generates the dates the bare alias generated in pandas 2."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            former = pd.date_range("2023-01-01", periods=6, freq=frequency)
        current = pd.date_range("2023-01-01", periods=6, freq=build_frequency_string(frequency, default_position="E"))
        assert list(current) == list(former)
