"""Tests for ``tsforecast.utils.abc.converter.TemporalConverter``.

``TemporalConverter`` is an abstract base class with no state of its own: the
contract under test is purely structural (abstract methods must be
implemented before instantiation) and is exercised through a minimal concrete
subclass defined in this module.
"""
from __future__ import annotations

import pytest

from tsforecast.utils.abc.converter import TemporalConverter


class _MinimalConverter(TemporalConverter):
    """Concrete subclass implementing only the two abstract methods."""

    def convert(self, value, from_unit, to_unit, **kwargs):
        return value * self.get_conversion_factor(from_unit, to_unit)

    def get_conversion_factor(self, from_unit, to_unit):
        # Facteur de conversion arbitraire (secondes <-> minutes) pour vérifier
        # que la classe concrète est bien appelée par convert()
        _seconds_per_unit = {"s": 1, "min": 60}
        return _seconds_per_unit[from_unit] / _seconds_per_unit[to_unit]


class TestTemporalConverterContract:
    """Structural contract enforced by the ``ABC``/``abstractmethod`` machinery."""

    def test_cannot_instantiate_base_class_directly(self):
        """The abstract class cannot be instantiated as is."""
        with pytest.raises(TypeError, match="abstract"):
            TemporalConverter()

    @pytest.mark.parametrize(
        "missing_method, class_body",
        [
            pytest.param(
                "get_conversion_factor",
                {"convert": lambda self, value, from_unit, to_unit, **kw: value},
                id="missing-get-conversion-factor",
            ),
            pytest.param(
                "convert",
                {"get_conversion_factor": lambda self, from_unit, to_unit: 1.0},
                id="missing-convert",
            ),
        ],
    )
    def test_partial_implementation_still_abstract(self, missing_method, class_body):
        """A subclass implementing only one of the two methods stays abstract.

        It cannot be instantiated.
        """
        partial_class = type("PartialConverter", (TemporalConverter,), class_body)
        with pytest.raises(TypeError, match=missing_method):
            partial_class()

    def test_full_implementation_is_instantiable(self):
        """A subclass implementing both abstract methods is instantiated normally."""
        converter = _MinimalConverter()
        assert isinstance(converter, TemporalConverter)

    def test_concrete_methods_are_used_as_defined_by_subclass(self):
        """``convert`` delegates to the subclass implementation of the conversion factor."""
        converter = _MinimalConverter()
        assert converter.get_conversion_factor("min", "s") == 60
        assert converter.convert(2, "min", "s") == 120

    def test_abstract_method_bodies_are_reachable_via_super(self):
        """Abstract method bodies stay reachable through ``super()`` from a subclass.

        The bodies (``pass``, hence ``None``) are exercised explicitly here,
        as ``TemporalNormalizer.validate`` does for its own body.
        """

        class _SuperCallingConverter(TemporalConverter):
            def convert(self, value, from_unit, to_unit, **kwargs):
                return super().convert(value, from_unit, to_unit, **kwargs)

            def get_conversion_factor(self, from_unit, to_unit):
                return super().get_conversion_factor(from_unit, to_unit)

        converter = _SuperCallingConverter()
        assert converter.convert(1, "s", "min") is None
        assert converter.get_conversion_factor("s", "min") is None
