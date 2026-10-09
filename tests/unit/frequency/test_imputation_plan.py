"""Tests for tsforecast.frequency.imputation_plan.

Focus §12.1 / §12.2 / §4.6 / §6.2 / §6.4 de [SPEC] high_frequency_imputer2_architecture.md :
étape immuable, couverture exacte de feature_cols par la voie de
matérialisation, invariant de repli, facteur d'échelle par ligne (pd.Series),
égalité Series-safe, provenance émise (souillures, repli, étape sans ancre),
plan immuable, vues (by_stage, models), sérialisation de diagnostic,
append_step, to_entity_tuple, INTERPOLATE_FALLBACK et MaterializationWay.
"""
# Modules de base
import dataclasses
import typing

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.imputation_plan import (
    ImputationStep,
    ImputationPlan,
    append_step,
    to_entity_tuple,
    INTERPOLATE_FALLBACK,
    MaterializationWay,
)
from tsforecast.frequency.provenance import ProvenanceType, resolve_model_provenance


# Fabrique d'étape : valeurs par défaut cohérentes, surchargeables au cas par cas
def _make_step(**overrides):
    """Build an ImputationStep with sensible defaults for the tests."""
    params = dict(
        pred_freq_label='M',
        pred_freq='M',
        var_key='gdp',
        var_name='gdp',
        model=LinearRegression(),
        feature_cols=('m1', 'q1'),
        scale_factor=3.0,
        fit_scale_factor=3.0,
        source_frequency='Q',
        entities=None,
        covariate_taint='none',
        target_taint='none',
        materialization={'m1': 'identity', 'q1': 'interpolate'},
        is_fallback=False,
        interpolation_method='linear',
        interpolation_anchor=1.0,
    )
    params.update(overrides)
    return ImputationStep(**params)


class TestImputationStep:
    """Étape : immuabilité et invariants de __post_init__."""

    def test_step_is_frozen(self):
        """Toute affectation sur une étape lève FrozenInstanceError."""
        step = _make_step()
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.var_name = 'autre'
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.is_fallback = True

    def test_materialization_must_cover_feature_cols(self):
        """Clé manquante et clé en trop lèvent ValueError en nommant la colonne."""
        # Clé manquante : 'q1' n'a pas de voie
        with pytest.raises(ValueError, match=r"q1"):
            _make_step(materialization={'m1': 'identity'})

        # Clé en trop : 'z9' n'est pas dans feature_cols
        with pytest.raises(ValueError, match=r"z9"):
            _make_step(
                materialization={'m1': 'identity', 'q1': 'interpolate', 'z9': 'identity'}
            )

    def test_unknown_materialization_way_raises(self):
        """Une voie hors des six littéraux du §4.6 lève ValueError."""
        with pytest.raises(ValueError, match=r"inconnues"):
            _make_step(materialization={'m1': 'identity', 'q1': 'teleport'})

    def test_fallback_invariant(self):
        """model is INTERPOLATE_FALLBACK avec is_fallback=False lève."""
        with pytest.raises(ValueError, match=r"is_fallback"):
            _make_step(
                model=INTERPOLATE_FALLBACK,
                feature_cols=(),
                materialization={},
                is_fallback=False,
            )

        # Cohérent : la sentinelle avec is_fallback=True passe
        step = _make_step(
            model=INTERPOLATE_FALLBACK,
            feature_cols=(),
            materialization={},
            is_fallback=True,
        )
        assert step.is_fallback is True
        assert step.emitted_provenance is ProvenanceType.INTERPOLATED

    def test_is_fallback_allowed_without_sentinel(self):
        """is_fallback=True reste permis avec un vrai estimateur (§6.4)."""
        step = _make_step(is_fallback=True)
        assert step.is_fallback is True
        assert step.emitted_provenance is ProvenanceType.INTERPOLATED

    def test_scale_factor_accepts_series(self):
        """Construction et égalité avec un scale_factor pd.Series (§5.4)."""
        idx = pd.date_range('2021-01-31', periods=3, freq='ME')
        scale = pd.Series([12.0, 3.0, 3.0], index=idx)

        step = _make_step(scale_factor=scale, fit_scale_factor=scale)
        assert isinstance(step.scale_factor, pd.Series)

        # Égalité tolérante aux Series : deux étapes équivalentes sont égales
        same = _make_step(
            model=step.model,
            scale_factor=pd.Series([12.0, 3.0, 3.0], index=idx),
            fit_scale_factor=pd.Series([12.0, 3.0, 3.0], index=idx),
        )
        assert step == same

        # Une Series différente casse l'égalité, sans lever
        other = _make_step(
            model=step.model,
            scale_factor=pd.Series([1.0, 1.0, 1.0], index=idx),
            fit_scale_factor=scale,
        )
        assert step != other

    def test_materialization_is_read_only_and_ordered(self):
        """materialization est gelée et rangée dans l'ordre de feature_cols."""
        step = _make_step(
            feature_cols=('q1', 'm1'),
            materialization={'m1': 'identity', 'q1': 'interpolate'},
        )
        assert list(step.materialization) == ['q1', 'm1']
        with pytest.raises(TypeError):
            step.materialization['m1'] = 'aggregate'

    def test_stage_key_property(self):
        """stage_key est le couple (pred_freq_label, var_key)."""
        step = _make_step(pred_freq_label='M', var_key='gdp')
        assert step.stage_key == ('M', 'gdp')


class TestImputationPlan:
    """Plan : conteneur immuable, groupement, vues, diagnostic."""

    def test_plan_is_immutable_and_append_returns_new(self):
        """append_step renvoie un nouveau plan sans muter l'ancien."""
        plan = ImputationPlan()
        assert len(plan) == 0

        step_a = _make_step(var_name='a1', var_key='a1')
        plan_2 = append_step(plan, step_a)

        # L'ancien plan est inchangé
        assert len(plan) == 0
        assert len(plan_2) == 1
        assert plan_2[0] is step_a

        # Le tuple de steps ne se réassigne pas
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan_2.steps = ()

        # Itération et indexation
        step_b = _make_step(var_name='a2', var_key='a2')
        plan_3 = append_step(plan_2, step_b)
        assert list(plan_3) == [step_a, step_b]
        assert plan_3[-1] is step_b

    def test_by_stage_preserves_order(self):
        """by_stage conserve l'ordre d'apparition des étapes et des groupes."""
        q_a1 = _make_step(pred_freq_label='Q', var_name='a1', var_key='a1')
        q_a2 = _make_step(pred_freq_label='Q', var_name='a2', var_key='a2')
        m_q1 = _make_step(pred_freq_label='M', var_name='q1', var_key='q1')
        m_a1 = _make_step(pred_freq_label='M', var_name='a1', var_key='a1')

        plan = ImputationPlan((q_a1, q_a2, m_q1, m_a1))
        by_stage = plan.by_stage()

        assert list(by_stage) == ['Q', 'M']
        assert by_stage['Q'] == (q_a1, q_a2)
        assert by_stage['M'] == (m_q1, m_a1)

    def test_models_view(self):
        """models() est la vue {stage_key: estimateur} du §13.2."""
        step = _make_step(pred_freq_label='M', var_key='gdp')
        plan = ImputationPlan((step,))
        assert plan.models() == {('M', 'gdp'): step.model}

    def test_diagnostic_frame_columns_and_emitted_provenance(self):
        """emitted_provenance vaut resolve_model_provenance sur trois souillures."""
        step_true = _make_step(
            var_name='v_true', var_key='v_true',
            covariate_taint='none', target_taint='none',
        )
        step_interp = _make_step(
            var_name='v_interp', var_key='v_interp',
            covariate_taint='interpolated', target_taint='none',
        )
        step_imputed = _make_step(
            var_name='v_imp', var_key='v_imp',
            covariate_taint='imputed', target_taint='imputed',
        )
        plan = ImputationPlan((step_true, step_interp, step_imputed))

        frame = plan.to_diagnostic_frame()
        assert list(frame.columns) == [
            'stage', 'variable', 'source_frequency', 'entities', 'n_entities',
            'scale_factor', 'fit_scale_factor', 'unanchored', 'n_features',
            'n_training_rows', 'n_written',
            'covariate_taint', 'target_taint',
            'emitted_provenance', 'is_fallback', 'interpolation_method',
            'interpolation_anchor', 'materialization', 'training_blocks',
        ]
        assert len(frame) == 3

        # La colonne emitted_provenance reproduit resolve_model_provenance
        assert frame.loc[0, 'emitted_provenance'] == resolve_model_provenance('none', 'none')
        assert frame.loc[1, 'emitted_provenance'] == resolve_model_provenance('interpolated', 'none')
        assert frame.loc[2, 'emitted_provenance'] == resolve_model_provenance('imputed', 'imputed')

        # materialization rendue lisible
        assert frame.loc[0, 'materialization'] == 'm1=identity, q1=interpolate'
        assert frame.loc[0, 'n_features'] == 2

    def test_diagnostic_frame_fallback_row(self):
        """Une étape en repli porte INTERPOLATED dans emitted_provenance."""
        fallback = _make_step(
            model=INTERPOLATE_FALLBACK, feature_cols=(), materialization={},
            is_fallback=True,
        )
        frame = ImputationPlan((fallback,)).to_diagnostic_frame()
        assert frame.loc[0, 'emitted_provenance'] is ProvenanceType.INTERPOLATED
        assert bool(frame.loc[0, 'is_fallback']) is True
        assert frame.loc[0, 'materialization'] == ''

    def test_empty_plan_diagnostic_frame(self):
        """Le plan vide produit un frame vide aux bonnes colonnes."""
        frame = ImputationPlan().to_diagnostic_frame()
        assert len(frame) == 0
        assert 'emitted_provenance' in frame.columns


# Les neuf couples de souillures, avec la provenance attendue (table du §6.3)
_TAINT_TABLE = [
    ('none', 'none', ProvenanceType.MODEL_ON_TRUE),
    ('none', 'interpolated', ProvenanceType.MODEL_ON_INTERPOLATED),
    ('none', 'imputed', ProvenanceType.MODEL_ON_IMPUTED_TARGET),
    ('interpolated', 'none', ProvenanceType.MODEL_ON_INTERPOLATED),
    ('interpolated', 'interpolated', ProvenanceType.MODEL_ON_INTERPOLATED),
    ('interpolated', 'imputed', ProvenanceType.MODEL_ON_IMPUTED_TARGET),
    ('imputed', 'none', ProvenanceType.MODEL_ON_IMPUTED),
    ('imputed', 'interpolated', ProvenanceType.MODEL_ON_IMPUTED),
    ('imputed', 'imputed', ProvenanceType.MODEL_ON_IMPUTED_BOTH),
]
_TAINT_IDS = [f'{c}-{t}' for c, t, _ in _TAINT_TABLE]


# =============================================================================
# ImputationStep — provenance émise
# =============================================================================
class TestEmittedProvenance:
    """``emitted_provenance``: taints, fallback and unanchored precedence."""

    @pytest.mark.parametrize('covariate_taint, target_taint, expected', _TAINT_TABLE, ids=_TAINT_IDS)
    def test_model_step_follows_the_taint_table(self, covariate_taint, target_taint, expected):
        """A model step emits the MODEL_* label of its two taints (§6.3)."""
        step = _make_step(covariate_taint=covariate_taint, target_taint=target_taint)
        assert step.emitted_provenance is expected

    @pytest.mark.parametrize('covariate_taint, target_taint, expected', _TAINT_TABLE, ids=_TAINT_IDS)
    def test_unanchored_primes_over_every_taint_combination(self, covariate_taint, target_taint, expected):
        """MODEL_UNANCHORED primes over the five MODEL_* labels, whatever the taints."""
        step = _make_step(
            covariate_taint=covariate_taint, target_taint=target_taint,
            unanchored=True, source_frequency=None, scale_factor=1.0, fit_scale_factor=1.0,
        )
        assert step.emitted_provenance is ProvenanceType.MODEL_UNANCHORED
        # Les deux souillures restent gelées dans l'étape, pour le diagnostic
        assert (step.covariate_taint, step.target_taint) == (covariate_taint, target_taint)

    @pytest.mark.parametrize('covariate_taint, target_taint, expected', _TAINT_TABLE, ids=_TAINT_IDS)
    def test_fallback_is_interpolated_whatever_the_taints(self, covariate_taint, target_taint, expected):
        """D6: a fallback step emits INTERPOLATED, never a MODEL_* label."""
        step = _make_step(
            covariate_taint=covariate_taint, target_taint=target_taint, is_fallback=True
        )
        assert step.emitted_provenance is ProvenanceType.INTERPOLATED

    @pytest.mark.parametrize('unanchored', [False, True], ids=['anchored', 'unanchored'])
    def test_fallback_primes_over_unanchored(self, unanchored):
        """A fallback produces interpolated cells, not model ones, even unanchored."""
        step = _make_step(is_fallback=True, unanchored=unanchored)
        assert step.emitted_provenance is ProvenanceType.INTERPOLATED

    @pytest.mark.parametrize('covariate_taint, target_taint, expected', _TAINT_TABLE, ids=_TAINT_IDS)
    def test_matches_resolve_model_provenance(self, covariate_taint, target_taint, expected):
        """On an anchored model step the property is ``resolve_model_provenance``."""
        step = _make_step(covariate_taint=covariate_taint, target_taint=target_taint)
        assert step.emitted_provenance is resolve_model_provenance(covariate_taint, target_taint)


# =============================================================================
# ImputationStep — valeurs par défaut, gel, clé d'étape
# =============================================================================
class TestImputationStepFields:
    """Defaults, frozen containers and stage key."""

    def test_diagnostic_fields_have_defaults(self):
        """Fit-time diagnostics default to empty / zero / False."""
        step = _make_step()
        assert dict(step.training_blocks) == {}
        assert step.unanchored is False
        assert step.n_training_rows == 0
        assert step.n_written == 0

    def test_feature_cols_list_is_normalized_to_a_tuple(self):
        """A list passed as ``feature_cols`` is stored as a tuple."""
        step = _make_step(feature_cols=['m1', 'q1'])
        assert step.feature_cols == ('m1', 'q1')

    def test_training_blocks_are_read_only(self):
        """``training_blocks`` is frozen like ``materialization``."""
        step = _make_step(training_blocks={('FR',): 'M', ('DE',): 'Q'})
        with pytest.raises(TypeError):
            step.training_blocks[('IT',)] = 'A'

    def test_training_blocks_do_not_alias_the_input_dict(self):
        """Mutating the dict given at construction does not change the step."""
        blocks = {('FR',): 'M'}
        step = _make_step(training_blocks=blocks)
        blocks[('DE',)] = 'Q'
        assert dict(step.training_blocks) == {('FR',): 'M'}

    def test_materialization_does_not_alias_the_input_dict(self):
        """Mutating the dict given at construction does not change the step."""
        materialization = {'m1': 'identity', 'q1': 'interpolate'}
        step = _make_step(materialization=materialization)
        materialization['m1'] = 'aggregate'
        assert step.materialization['m1'] == 'identity'

    def test_empty_feature_cols_with_empty_materialization(self):
        """A step with no feature (e.g. a fallback) has an empty materialization."""
        step = _make_step(feature_cols=(), materialization={}, is_fallback=True)
        assert step.feature_cols == ()
        assert dict(step.materialization) == {}

    def test_stage_key_for_a_time_series(self):
        """Time series: ``(frequency string, variable name)``."""
        step = _make_step(pred_freq_label='M', var_key='gdp')
        assert step.stage_key == ('M', 'gdp')

    def test_stage_key_for_a_panel_with_heterogeneous_frequency(self):
        """Panel: the label is a frozenset of (entity, freq) items, var_key a tuple."""
        label = frozenset({(('FR',), 'M'), (('DE',), 'Q')})
        step = _make_step(
            pred_freq_label=label, pred_freq={('FR',): 'M', ('DE',): 'Q'},
            var_key=('v', 'A'), entities=(('FR',), ('DE',)),
        )
        assert step.stage_key == (label, ('v', 'A'))
        # La clé est hachable : elle sert de clé de registre
        assert {step.stage_key: 1}[(label, ('v', 'A'))] == 1

    def test_unanchored_step_may_have_no_source_frequency(self):
        """Unanchored: ``source_frequency`` None and unit scale are accepted."""
        step = _make_step(unanchored=True, source_frequency=None, scale_factor=1.0, fit_scale_factor=1.0)
        assert step.source_frequency is None
        assert step.scale_factor == 1.0


# =============================================================================
# ImputationStep — égalité et hachage
# =============================================================================
class TestImputationStepEquality:
    """Field-by-field, Series-safe equality and hashing."""

    def test_step_equals_a_twin_sharing_the_model(self):
        """Same fields and same model object: equal."""
        model = LinearRegression()
        assert _make_step(model=model) == _make_step(model=model)

    def test_model_is_compared_by_identity(self):
        """Two distinct (identically configured) estimators make steps unequal."""
        assert _make_step(model=LinearRegression()) != _make_step(model=LinearRegression())

    @pytest.mark.parametrize(
        'override',
        [
            dict(pred_freq_label='Q'),
            dict(pred_freq='Q'),
            dict(var_key='other'),
            dict(var_name='other'),
            dict(feature_cols=('q1', 'm1')),
            dict(scale_factor=4.0),
            dict(fit_scale_factor=4.0),
            dict(source_frequency='A'),
            dict(entities=(('FR',),)),
            dict(covariate_taint='imputed'),
            dict(target_taint='imputed'),
            dict(materialization={'m1': 'identity', 'q1': 'aggregate'}),
            dict(is_fallback=True),
            dict(interpolation_method='cubic'),
            dict(interpolation_anchor=0.5),
            dict(training_blocks={('FR',): 'M'}),
            dict(unanchored=True),
            dict(n_training_rows=10),
            dict(n_written=5),
        ],
        ids=lambda o: next(iter(o)),
    )
    def test_each_field_participates_in_equality(self, override):
        """Changing any single field breaks equality."""
        model = LinearRegression()
        # feature_cols permuté : la matérialisation reste couverte (même ensemble)
        assert _make_step(model=model) != _make_step(model=model, **override)

    def test_comparison_with_another_type_is_not_equal(self):
        """Another type yields NotImplemented, hence ``==`` is False without raising."""
        step = _make_step()
        assert step.__eq__(3) is NotImplemented
        assert step != 3
        assert (step == 'M') is False

    def test_series_scale_factor_does_not_raise_on_equality(self):
        """No "truth value of a Series is ambiguous" error, for equal or unequal Series."""
        idx = pd.date_range('2021-01-31', periods=3, freq='ME')
        model = LinearRegression()
        left = _make_step(model=model, scale_factor=pd.Series([12.0, 3.0, 3.0], index=idx))
        equal = _make_step(model=model, scale_factor=pd.Series([12.0, 3.0, 3.0], index=idx))
        different = _make_step(model=model, scale_factor=pd.Series([12.0, 3.0, 4.0], index=idx))
        assert left == equal
        assert left != different

    def test_series_with_same_values_but_other_index_are_unequal(self):
        """Series equality is index-wise: a shifted index breaks equality."""
        model = LinearRegression()
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        left = _make_step(model=model, scale_factor=pd.Series([3.0, 3.0], index=idx))
        shifted = _make_step(
            model=model, scale_factor=pd.Series([3.0, 3.0], index=idx + pd.offsets.MonthEnd(1))
        )
        assert left != shifted

    def test_scalar_and_series_scale_factors_are_never_equal(self):
        """A scalar factor never equals a Series, even a constant one, in both orders."""
        model = LinearRegression()
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        scalar = _make_step(model=model, scale_factor=3.0)
        series = _make_step(model=model, scale_factor=pd.Series([3.0, 3.0], index=idx))
        assert scalar != series
        assert series != scalar

    def test_fit_scale_factor_series_is_compared_too(self):
        """``fit_scale_factor`` follows the same Series-safe comparison."""
        model = LinearRegression()
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        left = _make_step(model=model, fit_scale_factor=pd.Series([3.0, 3.0], index=idx))
        same = _make_step(model=model, fit_scale_factor=pd.Series([3.0, 3.0], index=idx))
        other = _make_step(model=model, fit_scale_factor=pd.Series([3.0, 6.0], index=idx))
        assert left == same
        assert left != other

    def test_equal_steps_have_equal_hashes(self):
        """The eq / hash contract: equal steps agree on their hash."""
        model = LinearRegression()
        assert hash(_make_step(model=model)) == hash(_make_step(model=model))

    def test_step_with_series_scale_is_hashable(self):
        """The hash ignores the (unhashable) Series fields."""
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        step = _make_step(scale_factor=pd.Series([3.0, 3.0], index=idx))
        assert isinstance(hash(step), int)

    def test_steps_are_usable_in_a_set(self):
        """Equal steps collapse in a set, distinct ones do not."""
        model = LinearRegression()
        steps = {
            _make_step(model=model), _make_step(model=model), _make_step(model=model, var_name='x'),
        }
        assert len(steps) == 2

    def test_replace_builds_a_variant_and_revalidates(self):
        """``dataclasses.replace`` builds a variant; invariants are re-checked."""
        step = _make_step()
        variant = dataclasses.replace(step, n_written=7)
        assert variant.n_written == 7 and step.n_written == 0
        with pytest.raises(ValueError, match=r"q1"):
            dataclasses.replace(step, materialization={'m1': 'identity'})


# =============================================================================
# INTERPOLATE_FALLBACK et MaterializationWay
# =============================================================================
class TestFallbackSentinelAndWays:
    """The fallback sentinel and the six materialization ways."""

    def test_sentinel_value(self):
        """The sentinel is the documented string."""
        assert INTERPOLATE_FALLBACK == 'interpolate_fallback'

    def test_another_string_model_is_not_the_sentinel(self):
        """Only the exact sentinel imposes ``is_fallback``."""
        step = _make_step(model='interpolate', feature_cols=(), materialization={}, is_fallback=False)
        assert step.is_fallback is False

    def test_sentinel_equal_string_imposes_is_fallback(self):
        """The check is by value: an equal string triggers the invariant."""
        with pytest.raises(ValueError, match='is_fallback'):
            _make_step(
                model='interpolate_fallback', feature_cols=(), materialization={}, is_fallback=False
            )

    def test_six_ways_in_rank_order(self):
        """The literal lists the six ways of §4.6, ordered by precedence rank."""
        assert typing.get_args(MaterializationWay) == (
            'identity', 'aggregate', 'stage_model', 'carried_model', 'interpolate', 'raw_anchors',
        )

    @pytest.mark.parametrize('way', typing.get_args(MaterializationWay))
    def test_every_way_is_accepted_by_a_step(self, way):
        """Each of the six ways is a valid materialization value."""
        step = _make_step(feature_cols=('m1',), materialization={'m1': way})
        assert step.materialization['m1'] == way

    @pytest.mark.parametrize('way', ['Identity', '', 'interpolate_fallback', None])
    def test_other_values_are_rejected(self, way):
        """Case variants, the sentinel and None are not ways."""
        with pytest.raises(ValueError, match='inconnues'):
            _make_step(feature_cols=('m1',), materialization={'m1': way})


# =============================================================================
# ImputationPlan — immuabilité, vues
# =============================================================================
class TestImputationPlanContainer:
    """Immutability, access protocol and views of the plan."""

    def test_list_input_is_frozen_into_a_tuple(self):
        """Steps given as a list are stored as a tuple, detached from the list."""
        steps = [_make_step(var_name='a1', var_key='a1')]
        plan = ImputationPlan(steps)
        steps.append(_make_step(var_name='a2', var_key='a2'))
        assert isinstance(plan.steps, tuple)
        assert len(plan) == 1

    def test_plan_is_frozen(self):
        """Reassigning ``steps`` raises FrozenInstanceError."""
        plan = ImputationPlan((_make_step(),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.steps = ()

    def test_len_iter_getitem_and_slice(self):
        """Sequence protocol: length, iteration, index, negative index, slice."""
        steps = [_make_step(var_name=name, var_key=name) for name in ('a', 'b', 'c')]
        plan = ImputationPlan(steps)
        assert len(plan) == 3
        assert list(plan) == steps
        assert plan[0] is steps[0] and plan[-1] is steps[2]
        # Une tranche rend un tuple d'étapes
        assert plan[1:] == tuple(steps[1:])

    def test_empty_plan(self):
        """The empty plan: no step, empty views."""
        plan = ImputationPlan()
        assert len(plan) == 0 and list(plan) == []
        assert plan.by_stage() == {}
        assert plan.models() == {}

    def test_plans_with_equal_steps_are_equal(self):
        """Dataclass equality delegates to the Series-safe step equality."""
        model = LinearRegression()
        assert ImputationPlan((_make_step(model=model),)) == ImputationPlan((_make_step(model=model),))
        assert ImputationPlan((_make_step(model=model),)) != ImputationPlan()

    def test_by_stage_with_panel_frozenset_labels(self):
        """Panel stages are keyed by their frozenset label, in order of appearance."""
        label_1 = frozenset({(('FR',), 'M'), (('DE',), 'Q')})
        label_2 = frozenset({(('FR',), 'M'), (('DE',), 'M')})
        s1 = _make_step(pred_freq_label=label_1, var_name='a', var_key='a')
        s2 = _make_step(pred_freq_label=label_2, var_name='a', var_key='a')
        s3 = _make_step(pred_freq_label=label_1, var_name='b', var_key='b')
        grouped = ImputationPlan((s1, s2, s3)).by_stage()
        assert list(grouped) == [label_1, label_2]
        assert grouped[label_1] == (s1, s3)

    def test_by_stage_returns_tuples(self):
        """Each group is an immutable tuple."""
        grouped = ImputationPlan((_make_step(),)).by_stage()
        assert all(isinstance(group, tuple) for group in grouped.values())

    def test_models_keys_are_stage_keys_in_plan_order(self):
        """``models()`` maps every stage key to its estimator, one entry per step."""
        s1 = _make_step(pred_freq_label='Q', var_name='a1', var_key='a1')
        s2 = _make_step(pred_freq_label='M', var_name='a1', var_key='a1')
        models = ImputationPlan((s1, s2)).models()
        assert list(models) == [('Q', 'a1'), ('M', 'a1')]
        assert models[('Q', 'a1')] is s1.model and models[('M', 'a1')] is s2.model

    def test_models_with_heterogeneous_panel_var_key(self):
        """A ``(variable, frequency)`` var_key is part of the registry key."""
        step = _make_step(var_key=('v', 'A'), var_name='v')
        assert ImputationPlan((step,)).models() == {('M', ('v', 'A')): step.model}

    def test_models_is_a_fresh_dict(self):
        """Mutating the returned registry does not alter the plan."""
        plan = ImputationPlan((_make_step(),))
        plan.models().clear()
        assert len(plan.models()) == 1


# =============================================================================
# to_diagnostic_frame
# =============================================================================
_DIAGNOSTIC_COLUMNS = [
    'stage', 'variable', 'source_frequency', 'entities', 'n_entities',
    'scale_factor', 'fit_scale_factor', 'unanchored', 'n_features',
    'n_training_rows', 'n_written',
    'covariate_taint', 'target_taint',
    'emitted_provenance', 'is_fallback', 'interpolation_method',
    'interpolation_anchor', 'materialization', 'training_blocks',
]


class TestDiagnosticFrame:
    """One row per step, exact columns, readable renderings."""

    @staticmethod
    def _panel_step(**overrides):
        params = dict(
            var_name='v', var_key='v',
            entities=(('FR',), ('DE',)),
            training_blocks={('FR',): 'M', ('DE',): 'Q'},
            n_training_rows=42, n_written=30,
            covariate_taint='interpolated', target_taint='imputed',
        )
        params.update(overrides)
        return _make_step(**params)

    def test_exact_columns_of_the_empty_plan(self):
        """The empty plan has the 19 documented columns, in order, and no row."""
        frame = ImputationPlan().to_diagnostic_frame()
        assert list(frame.columns) == _DIAGNOSTIC_COLUMNS
        assert len(frame) == 0

    def test_one_row_per_step_in_plan_order(self):
        """N steps give N rows, in the order of the plan."""
        steps = [_make_step(var_name=name, var_key=name) for name in ('c', 'a', 'b')]
        frame = ImputationPlan(steps).to_diagnostic_frame()
        assert list(frame.columns) == _DIAGNOSTIC_COLUMNS
        assert frame['variable'].tolist() == ['c', 'a', 'b']
        assert frame.index.tolist() == [0, 1, 2]

    def test_golden_row_of_a_time_series_step(self):
        """Every column of a time-series model step, against hand-written values."""
        step = _make_step(n_training_rows=12, n_written=36)
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['stage'] == 'M'
        assert row['variable'] == 'gdp'
        assert row['source_frequency'] == 'Q'
        # Série temporelle : entities None -> chaîne vide et 0 entité
        assert row['entities'] == ''
        assert row['n_entities'] == 0
        assert row['scale_factor'] == 3.0 and row['fit_scale_factor'] == 3.0
        assert bool(row['unanchored']) is False
        assert row['n_features'] == 2
        assert row['n_training_rows'] == 12 and row['n_written'] == 36
        assert row['covariate_taint'] == 'none' and row['target_taint'] == 'none'
        assert row['emitted_provenance'] is ProvenanceType.MODEL_ON_TRUE
        assert bool(row['is_fallback']) is False
        assert row['interpolation_method'] == 'linear'
        assert row['interpolation_anchor'] == 1.0
        assert row['materialization'] == 'm1=identity, q1=interpolate'
        assert row['training_blocks'] == ''

    def test_golden_row_of_a_panel_step(self):
        """Entities and training blocks are rendered as comma-separated strings."""
        row = ImputationPlan((self._panel_step(),)).to_diagnostic_frame().iloc[0]
        assert row['entities'] == 'FR, DE'
        assert row['n_entities'] == 2
        assert row['training_blocks'] == 'FR=M, DE=Q'
        assert row['emitted_provenance'] is ProvenanceType.MODEL_ON_IMPUTED_TARGET

    def test_multi_level_entity_label_is_joined_by_a_pipe(self):
        """A multi-level entity key is rendered ``level1|level2``."""
        step = _make_step(entities=(('EU', 'FR'),), training_blocks={('EU', 'FR'): 'M'})
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['entities'] == 'EU|FR'
        assert row['training_blocks'] == 'EU|FR=M'

    def test_degenerate_entity_of_a_time_series_in_training_blocks(self):
        """The ``()`` entity of a time series is rendered ``'()'``."""
        step = _make_step(training_blocks={(): 'M'})
        assert ImputationPlan((step,)).to_diagnostic_frame().loc[0, 'training_blocks'] == '()=M'

    def test_empty_entities_tuple_is_zero_entities(self):
        """An empty (non-None) entity tuple renders '' with n_entities 0."""
        step = _make_step(entities=())
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['entities'] == '' and row['n_entities'] == 0

    def test_per_row_scale_factor_is_rendered_per_row(self):
        """A Series factor is rendered 'per-row', the other factor stays numeric."""
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        step = _make_step(scale_factor=pd.Series([12.0, 3.0], index=idx), fit_scale_factor=3.0)
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['scale_factor'] == 'per-row'
        assert row['fit_scale_factor'] == 3.0

    def test_per_row_fit_scale_factor_is_rendered_per_row(self):
        """``fit_scale_factor`` has its own per-row rendering."""
        idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        step = _make_step(scale_factor=3.0, fit_scale_factor=pd.Series([12.0, 3.0], index=idx))
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['scale_factor'] == 3.0
        assert row['fit_scale_factor'] == 'per-row'

    def test_unanchored_step_row(self):
        """An unanchored step: flag, no source frequency, MODEL_UNANCHORED."""
        step = _make_step(
            unanchored=True, source_frequency=None, scale_factor=1.0, fit_scale_factor=1.0,
            covariate_taint='imputed', target_taint='imputed',
        )
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert bool(row['unanchored']) is True
        assert row['source_frequency'] is None
        assert row['emitted_provenance'] is ProvenanceType.MODEL_UNANCHORED
        # Les souillures calculées restent lisibles dans le diagnostic
        assert (row['covariate_taint'], row['target_taint']) == ('imputed', 'imputed')

    def test_fallback_row(self):
        """A fallback step: INTERPOLATED, no features, empty materialization."""
        step = _make_step(
            model=INTERPOLATE_FALLBACK, feature_cols=(), materialization={}, is_fallback=True,
        )
        row = ImputationPlan((step,)).to_diagnostic_frame().iloc[0]
        assert row['emitted_provenance'] is ProvenanceType.INTERPOLATED
        assert row['n_features'] == 0
        assert row['materialization'] == ''
        assert bool(row['is_fallback']) is True

    @pytest.mark.parametrize('covariate_taint, target_taint, expected', _TAINT_TABLE, ids=_TAINT_IDS)
    def test_emitted_provenance_column_follows_the_taint_table(self, covariate_taint, target_taint, expected):
        """The column reproduces the §6.3 table for each of the nine pairs."""
        step = _make_step(covariate_taint=covariate_taint, target_taint=target_taint)
        frame = ImputationPlan((step,)).to_diagnostic_frame()
        assert frame.loc[0, 'emitted_provenance'] is expected

    def test_materialization_follows_feature_cols_order(self):
        """The ``col=way`` rendering follows the order of ``feature_cols``."""
        step = _make_step(
            feature_cols=('q1', 'm1'),
            materialization={'m1': 'identity', 'q1': 'carried_model'},
        )
        frame = ImputationPlan((step,)).to_diagnostic_frame()
        assert frame.loc[0, 'materialization'] == 'q1=carried_model, m1=identity'

    def test_frame_is_a_fresh_object(self):
        """Mutating the frame does not alter the plan, and each call rebuilds it."""
        plan = ImputationPlan((_make_step(),))
        frame = plan.to_diagnostic_frame()
        frame.loc[0, 'variable'] = 'tampered'
        assert plan.to_diagnostic_frame().loc[0, 'variable'] == 'gdp'


# =============================================================================
# append_step et to_entity_tuple
# =============================================================================
class TestAppendStep:
    """Incremental construction without mutation."""

    def test_returns_a_new_plan_and_keeps_the_original(self):
        """The original plan is neither mutated nor returned."""
        original = ImputationPlan((_make_step(var_name='a', var_key='a'),))
        grown = append_step(original, _make_step(var_name='b', var_key='b'))
        assert grown is not original
        assert len(original) == 1 and len(grown) == 2

    def test_appended_step_is_last_and_identical(self):
        """The step is appended at the end, as the very same object."""
        first = _make_step(var_name='a', var_key='a')
        new = _make_step(var_name='b', var_key='b')
        grown = append_step(ImputationPlan((first,)), new)
        assert grown[0] is first and grown[1] is new

    def test_chaining_preserves_order(self):
        """Successive appends keep the order of insertion."""
        plan = ImputationPlan()
        for name in ('a', 'b', 'c'):
            plan = append_step(plan, _make_step(var_name=name, var_key=name))
        assert [step.var_name for step in plan] == ['a', 'b', 'c']

    def test_branching_from_the_same_plan(self):
        """Two appends on one plan give two independent plans."""
        base = ImputationPlan((_make_step(var_name='a', var_key='a'),))
        left = append_step(base, _make_step(var_name='l', var_key='l'))
        right = append_step(base, _make_step(var_name='r', var_key='r'))
        assert [s.var_name for s in left] == ['a', 'l']
        assert [s.var_name for s in right] == ['a', 'r']
        assert len(base) == 1


class TestToEntityTuple:
    """Freezing of the entity keys of a group."""

    def test_none_stays_none(self):
        """None flags a time series: it is kept."""
        assert to_entity_tuple(None) is None

    def test_list_becomes_a_tuple(self):
        """A list of keys becomes a tuple of the same keys."""
        assert to_entity_tuple([('France',), ('Italie',)]) == (('France',), ('Italie',))

    def test_empty_list_is_an_empty_tuple_not_none(self):
        """An empty group differs from a time series (None)."""
        result = to_entity_tuple([])
        assert result == () and result is not None

    def test_generator_is_consumed(self):
        """Any iterable is accepted."""
        assert to_entity_tuple(e for e in [('FR',), ('DE',)]) == (('FR',), ('DE',))

    def test_order_and_duplicates_are_kept(self):
        """No sorting and no deduplication."""
        assert to_entity_tuple([('b',), ('a',), ('b',)]) == (('b',), ('a',), ('b',))

    def test_result_feeds_a_step(self):
        """The tuple is directly storable in a step."""
        step = _make_step(entities=to_entity_tuple([('FR',)]))
        assert step.entities == (('FR',),)
