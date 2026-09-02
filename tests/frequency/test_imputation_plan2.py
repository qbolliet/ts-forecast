"""Tests for tsforecast.frequency.imputation_plan2.

Focus §12.1 / §4.6 / §6.2 de [SPEC] high_frequency_imputer2_architecture.md :
étape immuable v2, couverture exacte de feature_cols par la voie de
matérialisation, invariant de repli, facteur d'échelle par ligne (pd.Series),
plan immuable et sérialisation de diagnostic. Lot purement additif : la v1
(imputation_plan.ImputationStep) reste intacte.
"""
# Modules de base
import dataclasses

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.imputation_plan2 import (
    ImputationStep,
    ImputationPlan,
    append_step,
    INTERPOLATE_FALLBACK,
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
    """Étape v2 : immuabilité et invariants de __post_init__."""

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
        """stage_key reste le couple (pred_freq_label, var_key) de la v1."""
        step = _make_step(pred_freq_label='M', var_key='gdp')
        assert step.stage_key == ('M', 'gdp')


class TestImputationPlan:
    """Plan v2 : conteneur immuable, groupement, vues, diagnostic."""

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
            'stage', 'variable', 'n_features', 'covariate_taint', 'target_taint',
            'emitted_provenance', 'is_fallback', 'interpolation_method',
            'interpolation_anchor', 'materialization',
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


class TestV1Untouched:
    """La structure de plan de la v1 reste intacte (§12.2, §15.3)."""

    def test_v1_step_untouched(self):
        """imputation_plan.ImputationStep s'importe encore et porte trained_on_imputed."""
        from tsforecast.frequency.imputation_plan import ImputationStep as ImputationStepV1

        field_names = {f.name for f in dataclasses.fields(ImputationStepV1)}
        assert 'trained_on_imputed' in field_names
        # Les champs v2 ne se sont pas glissés dans la v1
        assert 'covariate_taint' not in field_names
        assert 'materialization' not in field_names

    def test_v1_and_v2_are_distinct_classes(self):
        """Les deux ImputationStep sont des classes distinctes, sans base commune."""
        from tsforecast.frequency.imputation_plan import ImputationStep as ImputationStepV1

        assert ImputationStep is not ImputationStepV1
        assert not issubclass(ImputationStep, ImputationStepV1)
        assert not issubclass(ImputationStepV1, ImputationStep)
