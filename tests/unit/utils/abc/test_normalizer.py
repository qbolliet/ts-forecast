"""Tests for ``tsforecast.utils.abc.normalizer.TemporalNormalizer``.

``TemporalNormalizer`` is an abstract base class exercised through a minimal
concrete subclass defined in this module. Its ``validate`` method is a
notable case: it is decorated ``@abstractmethod`` while already carrying a
concrete body (try/except around ``normalize``), so a subclass must still
override it explicitly, typically by delegating to ``super().validate()`` to
reuse that body.
"""
from __future__ import annotations

import pytest

from tsforecast.utils.abc.normalizer import TemporalNormalizer

_CODE_TO_LITERAL = {"D": "daily", "W": "weekly"}


class _MinimalNormalizer(TemporalNormalizer):
    """Concrete subclass implementing all three abstract methods."""

    def normalize(self, value):
        if value in _CODE_TO_LITERAL:
            return value
        reverse = self._build_reverse_mapping(_CODE_TO_LITERAL)
        if value in reverse:
            return reverse[value]
        raise ValueError(f"Unknown value: {value}")

    def to_literal(self, value):
        return _CODE_TO_LITERAL[self.normalize(value)]

    def validate(self, value):
        # Réutilisation du corps concret défini par la classe abstraite,
        # accolé au décorateur @abstractmethod
        return super().validate(value)


class TestTemporalNormalizerContract:
    """Structural contract enforced by the ``ABC``/``abstractmethod`` machinery."""

    def test_cannot_instantiate_base_class_directly(self):
        """La classe abstraite ne peut pas être instanciée telle quelle."""
        with pytest.raises(TypeError, match="abstract"):
            TemporalNormalizer()

    @pytest.mark.parametrize(
        "missing_methods, class_body",
        [
            pytest.param(
                ("to_literal", "validate"),
                {"normalize": lambda self, value: value},
                id="missing-to-literal-and-validate",
            ),
            pytest.param(
                ("normalize", "validate"),
                {"to_literal": lambda self, value: value},
                id="missing-normalize-and-validate",
            ),
            pytest.param(
                ("validate",),
                {
                    "normalize": lambda self, value: value,
                    "to_literal": lambda self, value: value,
                },
                id="missing-only-validate",
            ),
        ],
    )
    def test_partial_implementation_still_abstract(self, missing_methods, class_body):
        """Une sous-classe n'implémentant pas toutes les méthodes abstraites
        reste abstraite, y compris ``validate`` qui porte pourtant déjà un
        corps concret dans la classe de base.
        """
        partial_class = type("PartialNormalizer", (TemporalNormalizer,), class_body)
        with pytest.raises(TypeError) as exc_info:
            partial_class()
        for method_name in missing_methods:
            assert method_name in str(exc_info.value)

    def test_full_implementation_is_instantiable(self):
        """Une sous-classe implémentant les trois méthodes abstraites
        s'instancie normalement.
        """
        normalizer = _MinimalNormalizer()
        assert isinstance(normalizer, TemporalNormalizer)


class TestTemporalNormalizerConcreteBehavior:
    """Behavior of the concrete helpers exercised via ``_MinimalNormalizer``."""

    @pytest.fixture
    def normalizer(self):
        return _MinimalNormalizer()

    def test_normalize_resolves_code_and_literal(self, normalizer):
        """Code et littéral se normalisent tous deux vers le code."""
        assert normalizer.normalize("D") == "D"
        assert normalizer.normalize("daily") == "D"

    def test_normalize_unknown_value_raises(self, normalizer):
        """Une valeur non supportée lève une ``ValueError``."""
        with pytest.raises(ValueError, match="Unknown value"):
            normalizer.normalize("unknown")

    def test_to_literal_conversion(self, normalizer):
        """Conversion code -> littéral."""
        assert normalizer.to_literal("W") == "weekly"

    def test_validate_delegates_to_inherited_concrete_body(self, normalizer):
        """``validate`` (surchargée par simple délégation à ``super()``) réutilise
        le corps concret défini sur la méthode abstraite de la classe de base.
        """
        assert normalizer.validate("D") is True
        assert normalizer.validate("unknown") is False

    def test_build_reverse_mapping_is_inherited_unchanged(self):
        """La méthode statique utilitaire est héritée sans modification."""
        mapping = {"D": "daily", "W": "weekly"}
        assert TemporalNormalizer._build_reverse_mapping(mapping) == {
            "daily": "D",
            "weekly": "W",
        }

    def test_abstract_method_bodies_are_reachable_via_super(self):
        """Les corps des méthodes abstraites ``normalize``/``to_literal``
        (``pass``, donc ``None``) restent atteignables via ``super()`` depuis
        une sous-classe, à l'instar de ce que fait déjà ``validate``.
        """

        class _SuperCallingNormalizer(TemporalNormalizer):
            def normalize(self, value):
                return super().normalize(value)

            def to_literal(self, value):
                return super().to_literal(value)

            def validate(self, value):
                return super().validate(value)

        normalizer = _SuperCallingNormalizer()
        assert normalizer.normalize("D") is None
        assert normalizer.to_literal("D") is None
