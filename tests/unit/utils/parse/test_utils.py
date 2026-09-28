"""Tests for ``tsforecast.utils.parse.utils``.

This module covers the two only primitives that manipulate pandas frequency
strings as plain text: ``parse_frequency`` (splits a frequency string into its
base frequency, position and suffix components) and ``build_frequency_string``
(reassembles those components into a pandas-compatible frequency string).
Coverage includes the parse/build round trip for every supported frequency
family (calendar and sub-daily), anchored offsets, deliberate rejections
(pandas ``'A'``/``'AS'``/``'AE'`` aliases, an unhandled leading multiplier) and
casing sensitivity.
"""
from __future__ import annotations

import pytest

from tsforecast.utils.parse.utils import build_frequency_string, parse_frequency


class TestParseFrequency:
    """Contract of ``parse_frequency``: string -> (freq, position, suffix)."""

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
        """Base frequency, position and suffix are extracted correctly."""
        assert parse_frequency(frequency_str) == expected

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
        assert parse_frequency(frequency_str) == expected

    def test_none_input_raises_could_not_detect(self):
        """A ``None`` input (no frequency detected upstream) raises explicitly."""
        with pytest.raises(ValueError, match="Could not detect"):
            parse_frequency(None)

    def test_empty_string_raises_unable_to_parse(self):
        """An empty string does not match the expected grammar."""
        with pytest.raises(ValueError, match="Unable to parse"):
            parse_frequency("")

    def test_leading_multiplier_not_supported(self):
        """Comportement actuel à épingler : un multiplicateur en tête ('2MS')
        n'est pas géré par ``parse_frequency`` (non documenté comme supporté) :
        le chiffre initial fait échouer le motif de la grammaire attendue.
        """
        with pytest.raises(ValueError, match="Unable to parse"):
            parse_frequency("2MS")

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
        assert parse_frequency(frequency_str) == expected

    def test_case_sensitivity_lowercase_ms_is_not_month_start(self):
        """Casse significative : 'ms' (millisecondes) != 'MS' (début de mois)."""
        assert parse_frequency("ms") == ("ms", None, None)
        assert parse_frequency("MS") == ("M", "S", None)


class TestBuildFrequencyString:
    """Contract of ``build_frequency_string``: (freq, position, suffix) -> string."""

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
            pytest.param("B", "E", None, "BE", id="business-daily-end"),
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

    def test_suffix_kept_without_position(self):
        """Le suffixe est toujours accolé quand il est fourni, y compris sans
        position : nécessaire pour les ancres qui n'ont pas de notion de
        position S/E (ex : jour de la semaine, ``'W-MON'``) — cf.
        ``ANO-UTILS-001``, corrigée.
        """
        assert build_frequency_string("Q", position=None, suffix="DEC") == "Q-DEC"
        assert build_frequency_string("W", position=None, suffix="MON") == "W-MON"

    def test_suffix_kept_even_when_position_is_silently_ignored(self):
        """Comportement actuel à épingler : pour une fréquence non
        position-aware, la position demandée est ignorée mais le suffixe,
        lui, est tout de même accolé (chemin de code indépendant de la
        vérification ``_POSITION_AWARE_FREQUENCIES``).
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
        """Seuls 'S', 'E' et None sont des positions valides."""
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
        """Rejet explicite et délibéré des alias pandas 'A'/'AS'/'AE'
        (consolidation de ``parse_frequency``/``build_frequency_string`` comme
        seules primitives de fréquence, acceptée par l'auteur) : la fréquence
        de base est passée par ``normalize_frequency``, qui ne les reconnaît
        plus.
        """
        with pytest.raises(ValueError, match="Unsupported frequency"):
            build_frequency_string(frequency)

    def test_unsupported_base_frequency_raises(self):
        """Une fréquence de base inconnue est rejetée par la normalisation."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            build_frequency_string("not_a_frequency")

    def test_case_insensitive_base_frequency_normalization(self):
        """La fréquence de base passe par ``normalize_frequency`` : les noms
        littéraux (« monthly ») sont acceptés au même titre que les codes.
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
            "QE-DEC", "QS-JAN", "W-MON", "W-SUN",
        ],
    )
    def test_roundtrip_is_identity(self, frequency_str):
        """parse puis build reproduit exactement la chaîne d'origine."""
        freq, position, suffix = parse_frequency(frequency_str)
        assert build_frequency_string(freq, position=position, suffix=suffix) == frequency_str
