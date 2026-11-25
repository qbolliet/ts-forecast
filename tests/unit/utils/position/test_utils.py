"""Tests for ``tsforecast.utils.position.utils`` and the package ``__init__``.

Covers the module-level convenience functions: ``normalize_position``,
``to_literal``, ``to_code``, ``validate_position`` and ``flip_position``
(delegating to a shared ``PeriodPositionNormalizer``), ``convert_position``
and ``convert_offset`` (delegating to a fresh ``PeriodPositionConverter``,
imported lazily to avoid a circular import with ``tsforecast.utils.frequency``).
Equivalence with the class methods is the central property: these functions
add no logic of their own. The public surface re-exported by
``tsforecast/utils/position/__init__.py`` is checked as well.
"""
from __future__ import annotations

import pandas as pd
import pytest

import tsforecast.utils.position as position_package
from tsforecast.utils.position.converter import PeriodPositionConverter
from tsforecast.utils.position.normalizer import PeriodPositionNormalizer
from tsforecast.utils.position.utils import (
    convert_offset,
    convert_position,
    flip_position,
    normalize_position,
    to_code,
    to_literal,
    validate_position,
)

_VALID = ["S", "E", "start", "end"]


class TestPackageExports:
    """Public surface of ``tsforecast.utils.position``."""

    def test_all_lists_the_documented_symbols(self):
        """``__all__`` exposes the two classes, two types and seven functions."""
        assert sorted(position_package.__all__) == sorted([
            "PeriodPositionNormalizer", "PeriodPositionConverter",
            "PositionType", "UserPositionType",
            "normalize_position", "to_literal", "to_code", "validate_position",
            "flip_position", "convert_position", "convert_offset",
        ])

    @pytest.mark.parametrize("name", position_package.__all__)
    def test_every_exported_name_resolves(self, name):
        """Each name of ``__all__`` is an attribute of the package."""
        assert hasattr(position_package, name)

    def test_package_functions_are_the_utils_functions(self):
        """The package re-exports the very objects of ``utils.py`` (no wrapper)."""
        assert position_package.convert_offset is convert_offset


class TestNormalizationFunctions:
    """``normalize_position`` / ``to_code`` / ``to_literal`` / ``validate_position``."""

    @pytest.mark.parametrize("value", _VALID)
    def test_normalize_position_matches_class(self, value):
        """Same result as a dedicated ``PeriodPositionNormalizer``."""
        assert normalize_position(value) == PeriodPositionNormalizer().normalize(value)

    @pytest.mark.parametrize("value", _VALID)
    def test_to_code_matches_class(self, value):
        """Same result as ``PeriodPositionNormalizer.to_code``."""
        assert to_code(value) == PeriodPositionNormalizer().to_code(value)

    @pytest.mark.parametrize(
        "value, expected",
        [("S", "start"), ("E", "end"), ("start", "start"), ("end", "end")],
    )
    def test_to_literal_golden_values(self, value, expected):
        """Codes and literals both resolve to the literal name."""
        assert to_literal(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param("S", True, id="code"),
            pytest.param("end", True, id="literal"),
            pytest.param("End", False, id="wrong-case"),
            pytest.param("MS", False, id="pandas-offset"),
            pytest.param(None, False, id="none"),
            pytest.param(3, False, id="int"),
        ],
    )
    def test_validate_position(self, value, expected):
        """Boolean verdict, never an exception."""
        assert validate_position(value) is expected

    @pytest.mark.parametrize(
        "function",
        [normalize_position, to_code, to_literal, flip_position],
        ids=["normalize_position", "to_code", "to_literal", "flip_position"],
    )
    @pytest.mark.parametrize("invalid_value", ["middle", "s", None])
    def test_invalid_value_raises(self, function, invalid_value):
        """Every normalizing function rejects an unsupported position."""
        with pytest.raises(ValueError):
            function(invalid_value)


class TestFlipPosition:
    """``flip_position``: opposite code, involution."""

    @pytest.mark.parametrize(
        "value, expected", [("S", "E"), ("E", "S"), ("start", "E"), ("end", "S")]
    )
    def test_golden_values(self, value, expected):
        """Opposite position, always returned as a code."""
        assert flip_position(value) == expected

    @pytest.mark.parametrize("value", _VALID)
    def test_involution(self, value):
        """``flip(flip(x)) == normalize_position(x)``."""
        assert flip_position(flip_position(value)) == normalize_position(value)

    @pytest.mark.parametrize("value", _VALID)
    def test_matches_class(self, value):
        """Same result as ``PeriodPositionNormalizer.flip_position``."""
        assert flip_position(value) == PeriodPositionNormalizer().flip_position(value)


class TestConvertOffsetFunction:
    """``convert_offset`` delegates to ``PeriodPositionConverter.convert_offset``."""

    @pytest.mark.parametrize(
        "offset, position",
        [("MS", "end"), ("QE", "start"), ("2MS", "E"), ("D", "S"), ("YE-DEC", "start")],
    )
    def test_matches_class(self, offset, position):
        """Same result as a dedicated converter instance."""
        assert convert_offset(offset, position) == PeriodPositionConverter().convert_offset(offset, position)

    def test_error_propagated(self):
        """The converter's error on an invalid target position is propagated."""
        with pytest.raises(ValueError, match="Unsupported position"):
            convert_offset("MS", "middle")


class TestConvertPositionFunction:
    """``convert_position`` delegates to ``PeriodPositionConverter.convert``."""

    def test_matches_class_on_series(self):
        """Same result as ``PeriodPositionConverter().convert`` on a monthly series."""
        # Série MS de 3 mois : la conversion elle-même est testée dans test_converter.py,
        # seule l'équivalence fonction <-> méthode est vérifiée ici
        series = pd.Series(
            [1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"), name="x"
        )
        pd.testing.assert_series_equal(
            convert_position(series, "start", "end"),
            PeriodPositionConverter().convert(series, "start", "end"),
        )

    def test_freq_is_forwarded(self):
        """An explicit ``freq`` reaches the converter (a single date has no detectable frequency)."""
        # Une seule date : sans transmission de freq, la détection échouerait.
        # Valeur d'or : 2024-01-01 ouvre le 1er trimestre, qui se termine le 31/03
        result = convert_position(pd.DatetimeIndex(["2024-01-01"]), "S", "E", freq="Q")
        assert result[0] == pd.Timestamp("2024-03-31")

    def test_same_position_returns_input(self):
        """Identical source and target positions short-circuit the conversion."""
        index = pd.date_range("2024-01-01", periods=3, freq="MS")
        assert convert_position(index, "start", "S") is index
