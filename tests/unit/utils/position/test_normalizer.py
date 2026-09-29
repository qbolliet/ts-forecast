"""Tests for ``tsforecast.utils.position.normalizer.PeriodPositionNormalizer``.

Covers the public methods of ``PeriodPositionNormalizer``: ``normalize`` (and
its alias ``to_code``), ``to_literal``, ``validate`` and ``flip_position``,
plus its place in the ``TemporalNormalizer`` hierarchy and the ``PositionType``
/ ``UserPositionType`` literals it relies on. The class only knows four exact
spellings — codes ``'S'`` / ``'E'`` and literals ``'start'`` / ``'end'`` —
and is strictly case sensitive. Offset handling (``'MS'``, ``'QE-DEC'``...)
no longer belongs to the normalizer since the ``parse_frequency`` /
``build_frequency_string`` consolidation: those strings are rejected here.
"""
from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from tsforecast.utils.abc.normalizer import TemporalNormalizer
from tsforecast.utils.position.normalizer import (
    PeriodPositionNormalizer,
    PositionType,
    UserPositionType,
)

# Formes valides : codes et littéraux, dans le même ordre (S <-> start, E <-> end)
_CODES = ["S", "E"]
_LITERALS = ["start", "end"]

# Chaînes invalides : casse, quasi-littéraux, offsets pandas, alias abandonnés
_INVALID_STRINGS = [
    pytest.param("s", id="lowercase-code-s"),
    pytest.param("e", id="lowercase-code-e"),
    pytest.param("Start", id="titlecase-start"),
    pytest.param("END", id="uppercase-end"),
    pytest.param(" start", id="leading-space"),
    pytest.param("", id="empty-string"),
    pytest.param("middle", id="unknown-literal"),
    pytest.param("debut", id="french-literal"),
    pytest.param("MS", id="pandas-offset-not-a-position"),
    pytest.param("AS", id="abandoned-annual-alias"),
]

# Types non chaîne : rejet explicite par une ValueError dédiée
_NON_STRINGS = [
    pytest.param(None, id="none"),
    pytest.param(1, id="int"),
    pytest.param(np.nan, id="nan"),
    pytest.param(["S"], id="list"),
    pytest.param(("start",), id="tuple"),
]


@pytest.fixture
def normalizer() -> PeriodPositionNormalizer:
    """A fresh ``PeriodPositionNormalizer`` instance."""
    return PeriodPositionNormalizer()


class TestHierarchyAndTypes:
    """Place of the class in the abstract hierarchy and declared literals."""

    def test_is_a_temporal_normalizer(self, normalizer):
        """The class implements the ``TemporalNormalizer`` contract."""
        assert isinstance(normalizer, TemporalNormalizer)

    def test_position_type_literals(self):
        """``PositionType`` declares exactly the two codes, ``UserPositionType`` the two literals."""
        # Source de vérité des valeurs acceptées par normalize()
        assert (get_args(PositionType), get_args(UserPositionType)) == (
            ("S", "E"),
            ("start", "end"),
        )


class TestNormalize:
    """Contract of ``normalize``: any supported spelling -> position code."""

    @pytest.mark.parametrize("code", _CODES)
    def test_code_is_returned_unchanged(self, normalizer, code):
        """An already normalized code is returned as is."""
        assert normalizer.normalize(code) == code

    @pytest.mark.parametrize(
        "literal, expected_code", list(zip(_LITERALS, _CODES)), ids=_LITERALS
    )
    def test_literal_resolves_to_code(self, normalizer, literal, expected_code):
        """Each literal resolves to its code (``'start'`` -> ``'S'``, ``'end'`` -> ``'E'``)."""
        assert normalizer.normalize(literal) == expected_code

    @pytest.mark.parametrize("invalid_value", _INVALID_STRINGS)
    def test_unsupported_string_raises(self, normalizer, invalid_value):
        """Any other string is rejected, case variants included (no case folding).

        The message lists the accepted spellings, to guide the user.
        """
        with pytest.raises(ValueError, match=r"Unsupported position: .*Supported positions"):
            normalizer.normalize(invalid_value)

    @pytest.mark.parametrize("invalid_value", _NON_STRINGS)
    def test_non_string_raises_value_error(self, normalizer, invalid_value):
        """A non-string input is rejected by a dedicated ``ValueError`` (not a ``TypeError``)."""
        with pytest.raises(ValueError, match="Position must be a string"):
            normalizer.normalize(invalid_value)


class TestToCode:
    """``to_code`` is a strict alias of ``normalize``."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_matches_normalize_on_valid_values(self, normalizer, value):
        """Same result as ``normalize`` for every valid spelling."""
        assert normalizer.to_code(value) == normalizer.normalize(value)

    @pytest.mark.parametrize("invalid_value", ["middle", None])
    def test_same_error_as_normalize(self, normalizer, invalid_value):
        """Same exception and message as ``normalize`` for an invalid input."""
        with pytest.raises(ValueError) as from_normalize:
            normalizer.normalize(invalid_value)
        with pytest.raises(ValueError) as from_to_code:
            normalizer.to_code(invalid_value)
        assert str(from_to_code.value) == str(from_normalize.value)


class TestToLiteral:
    """Contract of ``to_literal``: any supported spelling -> literal name."""

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

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_roundtrip_through_literal_is_normalize(self, normalizer, value):
        """``to_code(to_literal(x)) == normalize(x)``: code <-> literal bijection."""
        assert normalizer.to_code(normalizer.to_literal(value)) == normalizer.normalize(value)

    def test_invalid_value_raises(self, normalizer):
        """The ``normalize`` error is propagated unchanged."""
        with pytest.raises(ValueError, match="Unsupported position"):
            normalizer.to_literal("middle")


class TestValidate:
    """``validate`` never raises; it wraps ``normalize`` in a try/except."""

    @pytest.mark.parametrize("value", _CODES + _LITERALS)
    def test_true_for_supported_values(self, normalizer, value):
        """Every value accepted by ``normalize`` is valid."""
        assert normalizer.validate(value) is True

    @pytest.mark.parametrize("invalid_value", _INVALID_STRINGS + _NON_STRINGS)
    def test_false_without_raising(self, normalizer, invalid_value):
        """Invalid strings and non-string types both yield ``False``, never an exception."""
        assert normalizer.validate(invalid_value) is False


class TestFlipPosition:
    """Contract of ``flip_position``: opposite position, always as a code."""

    @pytest.mark.parametrize(
        "position, expected",
        [
            pytest.param("S", "E", id="code-start"),
            pytest.param("E", "S", id="code-end"),
            pytest.param("start", "E", id="literal-start"),
            pytest.param("end", "S", id="literal-end"),
        ],
    )
    def test_returns_opposite_code(self, normalizer, position, expected):
        """Codes and literals are accepted; the output is always a code."""
        assert normalizer.flip_position(position) == expected

    @pytest.mark.parametrize("position", _CODES + _LITERALS)
    def test_is_an_involution_up_to_normalization(self, normalizer, position):
        """Flipping twice gives back the normalized input."""
        assert normalizer.flip_position(normalizer.flip_position(position)) == normalizer.normalize(position)

    @pytest.mark.parametrize("position", _CODES + _LITERALS)
    def test_never_returns_the_input_position(self, normalizer, position):
        """The flipped position always differs from the normalized input (no fixed point)."""
        assert normalizer.flip_position(position) != normalizer.normalize(position)

    @pytest.mark.parametrize("invalid_value", ["middle", "s", None])
    def test_invalid_value_raises(self, normalizer, invalid_value):
        """An invalid position is rejected instead of silently flipped to ``'S'``.

        The code falls back on ``'S'`` in its ``else`` branch: without the
        prior normalization, any unknown value would be "flipped" to the
        period start. This checks that the fallback is never reached.
        """
        with pytest.raises(ValueError):
            normalizer.flip_position(invalid_value)
