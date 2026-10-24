"""Tests of ``WarningAccumulator`` in ``tsforecast.imputation._engine.warnings``.

Scenarios: empty accumulator, exact text of each family for the fit and the transform,
repeats counted (degraded steps) or merged and sorted (unimputed pairs), emission order
and category, attribution of the warnings to the calling line.
"""

# Modules de base
import warnings

# Tests
import pytest

from tsforecast.imputation._engine.warnings import WarningAccumulator


# Émission enregistrée, filtres neutralisés
def _emitted(accumulator: WarningAccumulator) -> list:
    """Emit the accumulated warnings and return the recorded ones.

    Args:
        accumulator: Accumulator to emit.

    Returns:
        The recorded ``warnings.WarningMessage`` objects, in emission order.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        accumulator.emit()
    return list(caught)


class TestEmptyAccumulator:
    """Nothing recorded, nothing emitted."""

    @pytest.mark.parametrize('pass_name', ['fit', 'transform'])
    def test_no_message_and_no_warning(self, pass_name):
        accumulator = WarningAccumulator(pass_name)
        assert accumulator.messages() == []
        assert _emitted(accumulator) == []

    def test_families_start_empty(self):
        accumulator = WarningAccumulator('fit')
        assert accumulator.pass_name == 'fit'
        assert accumulator.degraded_steps == []
        assert accumulator.unimputed_pairs == []
        assert accumulator.outside_plan_cases == []


class TestDegradedSteps:
    """One message listing every degraded step, named after the pass."""

    @pytest.mark.parametrize('pass_name', ['fit', 'transform'])
    def test_exact_text(self, pass_name):
        accumulator = WarningAccumulator(pass_name)
        accumulator.add_degraded_step("'q1' at stage M: empty training set, interpolation fallback")
        accumulator.add_degraded_step("'a1' at stage Q: the model predicted only NaN, interpolation fallback")
        assert accumulator.messages() == [
            f"2 imputation step(s) degraded during the {pass_name}:\n"
            "  - 'q1' at stage M: empty training set, interpolation fallback\n"
            "  - 'a1' at stage Q: the model predicted only NaN, interpolation fallback"
        ]

    def test_repeats_are_counted(self):
        """Une étape dégradée deux fois compte deux fois, dans l'ordre d'arrivée."""
        accumulator = WarningAccumulator('transform')
        for message in ('b', 'a', 'b'):
            accumulator.add_degraded_step(message)
        assert accumulator.messages() == [
            "3 imputation step(s) degraded during the transform:\n  - b\n  - a\n  - b"
        ]


class TestUnimputedPairs:
    """One message naming each pair once, sorted by ``repr``, identical for both passes."""

    @pytest.mark.parametrize('pass_name', ['fit', 'transform'])
    def test_pairs_merged_and_sorted(self, pass_name):
        accumulator = WarningAccumulator(pass_name)
        accumulator.add_unimputed_pairs([('IT', 'v'), ('ES', 'v')])
        accumulator.add_unimputed_pairs([('IT', 'v')])
        assert accumulator.unimputed_pairs == [('IT', 'v'), ('ES', 'v'), ('IT', 'v')]
        assert accumulator.messages() == [
            "2 unobserved (entity, column) pair(s) could not be imputed: no "
            "prediction on the target grid (no usable covariate, or the prediction "
            "failed), and no interpolation fallback exists without an anchor; their "
            "cells stay NaN and unmarked: [('ES', 'v'), ('IT', 'v')]"
        ]

    def test_empty_iterable_records_nothing(self):
        accumulator = WarningAccumulator('fit')
        accumulator.add_unimputed_pairs([])
        assert accumulator.messages() == []


class TestOutsidePlanCases:
    """One message listing every case outside the frozen plan."""

    def test_exact_text(self):
        accumulator = WarningAccumulator('transform')
        accumulator.add_outside_plan_case(
            "1 entity/entities absent from the fit (('ES',),) are left untouched "
            "(impute_unobserved_entities=False)"
        )
        assert accumulator.messages() == [
            "1 case(s) of entities outside the frozen plan at transform time:\n"
            "  - 1 entity/entities absent from the fit (('ES',),) are left untouched "
            "(impute_unobserved_entities=False)"
        ]


class TestEmission:
    """One ``UserWarning`` per non-empty family, always in the same order, attributed to the caller."""

    def test_order_and_category(self):
        accumulator = WarningAccumulator('transform')
        # Enregistrement dans l'ordre inverse de l'émission
        accumulator.add_outside_plan_case('case')
        accumulator.add_unimputed_pairs([('IT', 'v')])
        accumulator.add_degraded_step('step')

        caught = _emitted(accumulator)

        assert [item.category for item in caught] == [UserWarning] * 3
        assert [str(item.message) for item in caught] == accumulator.messages()
        assert str(caught[0].message).startswith('1 imputation step(s) degraded')
        assert str(caught[1].message).startswith('1 unobserved (entity, column) pair(s)')
        assert str(caught[2].message).startswith('1 case(s) of entities outside')

    def test_empty_families_are_skipped(self):
        accumulator = WarningAccumulator('fit')
        accumulator.add_unimputed_pairs([('IT', 'v')])
        caught = _emitted(accumulator)
        assert len(caught) == 1
        assert 'unobserved' in str(caught[0].message)

    def test_warning_is_attributed_to_the_calling_line(self):
        """L'emplacement est celui de l'appelant : module et filtres de l'imputeur."""
        accumulator = WarningAccumulator('fit')
        accumulator.add_degraded_step('step')
        caught = _emitted(accumulator)
        assert caught[0].filename == __file__

    def test_messages_do_not_consume_the_accumulator(self):
        """Rendre les messages ne vide rien : ``emit`` les émet ensuite tels quels."""
        accumulator = WarningAccumulator('fit')
        accumulator.add_degraded_step('step')
        first = accumulator.messages()
        assert accumulator.messages() == first
        assert [str(item.message) for item in _emitted(accumulator)] == first
