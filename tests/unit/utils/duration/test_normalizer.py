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
        """Un code déjà normalisé est renvoyé tel quel."""
        assert normalizer.normalize(code) == code

    @pytest.mark.parametrize(
        "literal, expected_code",
        list(zip(_LITERALS, _CODES)),
        ids=_LITERALS,
    )
    def test_literal_resolves_to_code(self, normalizer, literal, expected_code):
        """Chaque nom littéral se résout vers son code correspondant."""
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
        """Repli sur ``parse_frequency`` : la position et le suffixe d'ancrage
        d'une chaîne de fréquence pandas complète sont ignorés, seule la
        fréquence de base est retenue.
        """
        assert normalizer.normalize(frequency_str) == expected_code

    def test_uppercase_s_is_not_recognized_as_seconds(self, normalizer):
        """Piège epinglé (notebook duration_normalizer) : le code des secondes
        est ``'s'`` minuscule. ``'S'`` majuscule n'est ni un code ni un
        littéral connu, et le repli via ``parse_frequency('S')`` extrait une
        base identique à la valeur d'entrée : le garde-fou anti-boucle infinie
        empêche la récursion, la normalisation échoue donc au lieu de
        retomber sur la seconde.
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
        """Aucune normalisation de casse n'est appliquée : une variante mal
        capitalisée d'un code ou d'un littéral pourtant valide est rejetée.
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
        """Un type non ``str`` est explicitement rejeté (pas de ``TypeError``
        laissé remonter naturellement, contrairement à ``DurationConverter``).
        """
        with pytest.raises(ValueError, match="must be a string"):
            normalizer.normalize(invalid_value)

    @pytest.mark.parametrize(
        "invalid_value",
        [pytest.param("xyz", id="unknown-code"), pytest.param("", id="empty-string")],
    )
    def test_unsupported_string_raises(self, normalizer, invalid_value):
        """Une chaîne non reconnue, même après repli sur ``parse_frequency``,
        lève une erreur explicite.
        """
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.normalize(invalid_value)


class TestToCode:
    """``to_code`` is a strict alias of ``normalize``."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_matches_normalize_on_valid_values(self, normalizer, value):
        """Résultat identique à ``normalize`` pour toute entrée valide."""
        assert normalizer.to_code(value) == normalizer.normalize(value)

    def test_matches_normalize_error_on_invalid_value(self, normalizer):
        """Même erreur que ``normalize`` pour une entrée invalide."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.to_code("xyz")


class TestToLiteral:
    """Contract of ``to_literal``: any supported representation -> literal."""

    @pytest.mark.parametrize(
        "code, expected_literal", list(zip(_CODES, _LITERALS)), ids=_CODES
    )
    def test_code_resolves_to_literal(self, normalizer, code, expected_literal):
        """Un code se résout vers son nom littéral."""
        assert normalizer.to_literal(code) == expected_literal

    @pytest.mark.parametrize("literal", _LITERALS)
    def test_literal_is_idempotent(self, normalizer, literal):
        """Un littéral déjà normalisé se résout vers lui-même."""
        assert normalizer.to_literal(literal) == literal

    @pytest.mark.parametrize("code", _CODES)
    def test_roundtrip_to_code_to_literal_is_identity(self, normalizer, code):
        """``to_code(to_literal(code)) == code`` : bijection code <-> littéral."""
        assert normalizer.to_code(normalizer.to_literal(code)) == code

    def test_invalid_value_raises(self, normalizer):
        """L'erreur de ``normalize`` est propagée telle quelle."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            normalizer.to_literal("xyz")


class TestValidate:
    """``validate`` never raises; it wraps ``normalize`` in a try/except."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS + ["MS", "QE-DEC", "W-MON"])
    def test_true_for_supported_values(self, normalizer, value):
        """Toute valeur acceptée par ``normalize`` est validée."""
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
        """Aucune exception ne remonte, y compris pour des types manifestement
        invalides : ``validate`` renvoie toujours un booléen.
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
        """L'ordre déclaré (ns < us < ... < B < W < SM < M < Q < Y) est
        respecté entre deux codes consécutifs, dans les deux sens.
        """
        assert normalizer.is_longer_duration(longer, shorter) is True
        assert normalizer.is_longer_duration(shorter, longer) is False

    def test_mixed_code_and_literal_formats(self, normalizer):
        """Les deux arguments peuvent mélanger code et littéral."""
        assert normalizer.is_longer_duration("month", "day") is True
        assert normalizer.is_longer_duration("M", "D") is True
        assert normalizer.is_longer_duration("month", "D") is True
        assert normalizer.is_longer_duration("day", "month") is False

    @pytest.mark.parametrize("code", _CODES)
    def test_equality_is_always_false(self, normalizer, code):
        """Comparaison stricte (``>``) : ``is_longer_duration(x, x)`` est
        toujours ``False``, jamais ``>=``.
        """
        assert normalizer.is_longer_duration(code, code) is False

    def test_total_order_is_transitive(self, normalizer):
        """Propriété : la relation définit un ordre total sur l'ensemble des
        codes (antisymétrie et transitivité), cohérent avec l'ordre déclaré
        dans ``_CODES``.
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
        """``is_longer_duration`` appelle ``to_code`` sur ses deux arguments,
        qui lève une ``ValueError`` avant d'atteindre le repli interne
        ``_duration_order.get(code, 0)`` : une durée invalide n'est donc
        jamais traitée silencieusement comme "la plus courte".
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
        """Équivalence stricte avec ``validate(dur1) and validate(dur2)`` :
        aucune notion de rapport de conversion raisonnable entre les deux
        durées n'est vérifiée (ex : ``'ns'``/``'Y'`` sont jugées "compatibles").
        """
        expected = normalizer.validate(dur1) and normalizer.validate(dur2)
        assert normalizer.are_compatible_durations(dur1, dur2) == expected
