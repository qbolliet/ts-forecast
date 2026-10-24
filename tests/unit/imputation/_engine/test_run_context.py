"""Tests of ``FittedComponents`` and ``_RunContext`` in ``tsforecast.imputation._engine.run_context``.

Scenarios: configuration of the shared components, one fresh materializer per pass with
independent registers, wiring of the fit context (training-set builder on the materializer
of the pass, training mask and log injected, no aggregation constraint on the builder),
replay context built from the fitted components alone, isolation of two contexts.
"""

# Modules de base
import dataclasses

# Manipulation de données
import pandas as pd

# Tests
import pytest

from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.imputation._components.provenance import ImputationProvenanceTracker
from tsforecast.imputation._components.variable_orderer import VariableOrderer
from tsforecast.imputation._engine.run_context import FittedComponents, _RunContext

# Contrainte par colonne : sa transmission se vérifie par identité
_CONSTRAINT = {'a1': None, '__default__': 'sum'}
# Fréquences par colonne d'un panel, une colonne à fréquence par entité
_COLUMN_FREQUENCIES = {'m1': 'M', 'v': {('FR',): 'Y', ('DE',): 'Q'}}


# Composants construits avec des paramètres non triviaux
def _build_components(**overrides) -> FittedComponents:
    """Build components from representative parameters, overriding some of them.

    Args:
        **overrides: Keyword arguments of :meth:`FittedComponents.build` to change.

    Returns:
        The components.
    """
    params = dict(
        covariate_strategy='model',
        covariate_fallback='interpolate',
        covariate_eligibility='any_entity',
        interpolation_method={'v': 'linear', '__default__': 'linear'},
        interpolation_anchor=None,
        aggregation_constraint=_CONSTRAINT,
        scale_features='calendar',
        column_frequencies=_COLUMN_FREQUENCIES,
    )
    params.update(overrides)
    return FittedComponents.build(**params)


# Contexte de fit sur des collaborateurs neufs
def _fit_context(components: FittedComponents, **overrides) -> _RunContext:
    """Build a fit context with fresh collaborators.

    Args:
        components: Shared components.
        **overrides: Keyword arguments of :meth:`_RunContext.for_fit` to change.

    Returns:
        The context.
    """
    params = dict(
        window_calculator=ImputationWindowCalculator(),
        provenance_tracker=ImputationProvenanceTracker(),
        variable_orderer=VariableOrderer(),
        training_mask=lambda frequencies: None,
    )
    params.update(overrides)
    return _RunContext.for_fit(components, **params)


# Contexte de rejeu sur des collaborateurs neufs
def _transform_context(components: FittedComponents, **overrides) -> _RunContext:
    """Build a replay context with fresh collaborators.

    Args:
        components: Shared components.
        **overrides: Keyword arguments of :meth:`_RunContext.for_transform` to change.

    Returns:
        The context.
    """
    params = dict(
        window_calculator=ImputationWindowCalculator(),
        provenance_tracker=ImputationProvenanceTracker(),
        transform_frequencies={},
        transform_misaligned=(),
    )
    params.update(overrides)
    return _RunContext.for_transform(components, **params)


class TestFittedComponentsBuild:
    """The shared components carry the parameters of the imputer, untouched."""

    def test_aggregation_constraint_configuration(self):
        components = _build_components()
        constraint = components.aggregation_constraint
        assert constraint.aggregation_constraint is _CONSTRAINT
        assert constraint.context == 'HighFrequencyImputer'

    def test_stage_scaler_configuration(self):
        components = _build_components()
        assert components.stage_scaler.scale_features == 'calendar'
        assert components.stage_scaler.column_frequencies is _COLUMN_FREQUENCIES

    def test_materializer_options(self):
        components = _build_components()
        assert dict(components.materializer_options) == {
            'covariate_strategy': 'model',
            'covariate_fallback': 'interpolate',
            'covariate_eligibility': 'any_entity',
            'interpolation_method': {'v': 'linear', '__default__': 'linear'},
            'interpolation_anchor': None,
            'aggregation_constraint': _CONSTRAINT,
        }

    def test_components_are_frozen(self):
        components = _build_components()
        with pytest.raises(dataclasses.FrozenInstanceError):
            components.stage_scaler = None


class TestNewMaterializer:
    """One materializer per call: configured alike, registers of its own."""

    def test_fresh_instance_with_empty_registers(self):
        components = _build_components()
        first, second = components.new_materializer(), components.new_materializer()
        assert first is not second
        for materializer in (first, second):
            assert materializer.imputed_store == {}
            assert materializer.imputed_freq_store == {}
            assert materializer.origin_store == {}

    def test_configuration_and_shared_applier(self):
        """Le matérialiseur porte la contrainte, et recale par l'objet partagé."""
        components = _build_components()
        materializer = components.new_materializer()
        assert materializer.covariate_strategy == 'model'
        assert materializer.covariate_fallback == 'interpolate'
        assert materializer.covariate_eligibility == 'any_entity'
        assert materializer.aggregation_constraint is _CONSTRAINT
        assert materializer.aggregation_constraint_applier is components.aggregation_constraint

    def test_registers_are_independent(self):
        components = _build_components()
        first, second = components.new_materializer(), components.new_materializer()
        first.imputed_store['v'] = pd.Series([1.0])
        assert second.imputed_store == {}

    def test_options_are_validated_at_each_materializer(self):
        """Une option refusée lève à la construction du matérialiseur."""
        components = _build_components(covariate_strategy='unknown')
        with pytest.raises(ValueError, match='covariate_strategy'):
            components.new_materializer()


class TestForFit:
    """The fit context: fresh materializer, builder wired on it, empty accumulators."""

    def test_training_set_builder_is_wired_on_the_materializer_of_the_pass(self):
        components = _build_components()

        def mask(frequencies):
            return None

        def log(message):
            return None

        run = _fit_context(components, training_mask=mask, log=log)
        builder = run.training_set_builder
        assert builder.materializer is run.materializer
        assert builder.training_mask is mask
        assert builder.log is log
        # Le builder ne porte aucune contrainte : la cible n'est jamais agrégée
        assert not hasattr(builder, 'aggregation_constraint')

    def test_collaborators_and_shared_components(self):
        components = _build_components()
        calculator = ImputationWindowCalculator()
        tracker = ImputationProvenanceTracker()
        orderer = VariableOrderer()
        run = _fit_context(
            components,
            window_calculator=calculator,
            provenance_tracker=tracker,
            variable_orderer=orderer,
        )
        assert run.window_calculator is calculator
        assert run.provenance_tracker is tracker
        assert run.variable_orderer is orderer
        assert run.aggregation_constraint is components.aggregation_constraint
        assert run.stage_scaler is components.stage_scaler
        assert run.materializer.aggregation_constraint_applier is components.aggregation_constraint

    def test_accumulators_start_empty(self):
        run = _fit_context(_build_components())
        assert run.pending_warnings.pass_name == 'fit'
        assert run.pending_warnings.messages() == []
        assert run.unanchored_written == set()
        assert run.transform_frequencies == {}
        assert run.transform_misaligned == frozenset()
        # Sans journalisation demandée, le builder n'en reçoit aucune
        assert run.training_set_builder.log is None


class TestForTransform:
    """The replay context: built from the fitted components alone, nothing of the fit."""

    def test_shared_components_and_fresh_materializer(self):
        components = _build_components()
        fit_run = _fit_context(components)
        # Productions du fit dans ses registres : le rejeu n'en voit rien
        fit_run.materializer.imputed_store['v'] = pd.Series([1.0])

        run = _transform_context(components)

        assert run.aggregation_constraint is components.aggregation_constraint
        assert run.stage_scaler is components.stage_scaler
        assert run.materializer is not fit_run.materializer
        assert run.materializer.imputed_store == {}
        assert run.materializer.aggregation_constraint_applier is components.aggregation_constraint

    def test_no_training_set_builder_nor_orderer(self):
        run = _transform_context(_build_components())
        assert run.training_set_builder is None
        assert run.variable_orderer is None
        assert run.pending_warnings.pass_name == 'transform'
        assert run.unanchored_written == set()

    def test_collections_are_copied(self):
        frequencies = {('ES', 'q1'): 'Q'}
        misaligned = [('ES', 'a1')]
        run = _transform_context(
            _build_components(),
            transform_frequencies=frequencies,
            transform_misaligned=misaligned,
        )
        frequencies[('ES', 'v')] = 'Y'
        misaligned.append(('ES', 'v'))
        assert run.transform_frequencies == {('ES', 'q1'): 'Q'}
        assert run.transform_misaligned == frozenset({('ES', 'a1')})

    def test_two_contexts_share_no_mutable_state(self):
        """Deux rejeux successifs partent chacun d'un état vierge."""
        components = _build_components()
        first, second = _transform_context(components), _transform_context(components)

        first.materializer.imputed_store['v'] = pd.Series([1.0])
        first.pending_warnings.add_degraded_step('step')
        first.unanchored_written.add(('ES', 'v'))
        first.transform_frequencies[('ES', 'v')] = 'Y'

        assert second.materializer.imputed_store == {}
        assert second.pending_warnings.messages() == []
        assert second.unanchored_written == set()
        assert second.transform_frequencies == {}
