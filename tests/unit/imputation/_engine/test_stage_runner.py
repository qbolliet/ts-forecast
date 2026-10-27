"""Tests of ``StageRunner``, ``StageParams`` and the step helpers of ``tsforecast.imputation._engine.stage_runner``.

Scenarios: the prediction grid of the pass on PANEL-F; covariate selection, empty training rows,
estimator resolution, group scale factor, pairs of a step; the parameters (every field required,
the imputer's own values passed through); composition of a (stage, variable) on PANEL-F (51
pooled rows, golden scaled targets at stages M and Q, no register written); the fit of a stage (one
fit per (stage, variable) shared by its groups, order and CV scores, fallback, degraded step,
unanchored groups, replay context refused, plan identical to the imputer's); the execution of a step
(writes, empty grid, replay delegation); and the single execution path of the fit and of the replay
on PANEL-X (call counts, neutralized execution, identical prediction frames).
"""
# Modules de base
import warnings
from dataclasses import fields, replace
from functools import partial
from typing import List, Tuple

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest

from tests.support.estimators import FailingEstimator, SpyEstimator
from tsforecast.imputation import HighFrequencyImputer, ProvenanceType
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer
from tsforecast.imputation._components.imputation_plan import (
    INTERPOLATE_FALLBACK,
    ImputationPlan,
    append_step,
)
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.imputation._components.provenance import ImputationProvenanceTracker
from tsforecast.imputation._components.stage_scaler import StageScaler
from tsforecast.imputation._components.variable_orderer import VariableOrderer
from tsforecast.imputation._engine.preparation import prepare_work_frame
from tsforecast.imputation._engine.run_context import FittedComponents, _RunContext
from tsforecast.imputation._engine.grids import training_mask_at
from tsforecast.imputation._engine.stage_runner import (
    StageOutcome,
    StageParams,
    StageRunner,
    drop_empty_training_rows,
    estimator_for,
    group_scale_factor,
    select_feature_columns,
    step_pairs,
)

# Clés d'entité des jeux de référence
FR, DE, IT = ('FR',), ('DE',), ('IT',)
# Provenances qu'écrit l'exécution d'une étape
_STEP_PROVENANCES = {
    member.value for member in ProvenanceType
    if member.value.startswith('model') or member is ProvenanceType.INTERPOLATED
}


# =============================================================================
# Doublures et constructeurs
# =============================================================================
# Estimateur dont l'ajustement réussit et dont la prédiction échoue
class _PredictFailingEstimator(SpyEstimator):
    """Spy estimator whose ``fit`` succeeds and whose ``predict`` always raises."""

    def predict(self, X):
        """Always raise, to exercise the degradation at execution time."""
        raise RuntimeError('deliberate predict failure')


# Estimateur qui ne prédit que des NaN
class _NaNPredictingEstimator(SpyEstimator):
    """Spy estimator whose predictions are all NaN."""

    def predict(self, X):
        """Record the frame, then predict NaN on every row."""
        super().predict(X)
        return np.full(len(X), np.nan)


# Estimateur dont la première prédiction est NaN
class _FirstNaNEstimator(SpyEstimator):
    """Spy estimator whose first prediction of each call is NaN."""

    def predict(self, X):
        """Predict the learnt mean, except on the first row."""
        values = super().predict(X).astype(float)
        values[:1] = np.nan
        return values


# Calculateur de fenêtre minimal : un masque donné, des appels enregistrés
class _FixedMaskCalculator:
    """Window calculator double returning one given mask and recording its calls."""

    def __init__(self, mask: pd.Series) -> None:
        self.mask = mask
        self.calls = []

    def get_mask_at_frequency(self, frequency, kind='imputation'):
        """Record the call and return the given mask."""
        self.calls.append((frequency, kind))
        return self.mask


# Imputeur non ajusté, estimateur espion par défaut
def _imputer(**overrides) -> HighFrequencyImputer:
    """Build an unfitted imputer targeting the monthly frequency.

    Args:
        **overrides: Constructor parameters replacing the defaults
            ``target_frequency='M'`` and ``estimator=SpyEstimator()``.

    Returns:
        The imputer.
    """
    params = dict(target_frequency='M', estimator=SpyEstimator())
    params.update(overrides)
    return HighFrequencyImputer(**params)


# Exécuteur d'un fit construit à partir des briques publiques, avant la phase 5
def _fit_runner(
    imputer: HighFrequencyImputer,
    data: pd.DataFrame,
    *,
    log=None,
) -> Tuple[StageRunner, pd.DataFrame]:
    """Build the stage runner of a fit the way the imputer wires it, before any stage runs.

    Args:
        imputer: Unfitted imputer, read for its parameters only.
        data: Fit data.
        log: Logging callback of the runner.

    Returns:
        Tuple ``(runner, X_work)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = prepare_work_frame(imputer, data, fitting=True)
    layout = prepared.fit.layout
    components = FittedComponents.build(
        covariate_strategy=imputer.covariate_strategy,
        covariate_fallback=imputer.covariate_fallback,
        covariate_eligibility=imputer.covariate_eligibility,
        interpolation_method=imputer.interpolation_method,
        interpolation_anchor=imputer.interpolation_anchor,
        aggregation_constraint=imputer.aggregation_constraint,
        scale_features=imputer.scale_features,
        column_frequencies=layout.frequencies_by_column(),
    )
    orderer = VariableOrderer(
        fit_predict_order=imputer.fit_predict_order,
        cv=imputer.cv,
        cv_scoring=imputer.cv_scoring,
        min_cv_train_size=imputer.min_cv_train_size,
    )
    if imputer.fit_predict_order == 'cv':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            orderer.fit()
    run = _RunContext.for_fit(
        components,
        window_calculator=prepared.window_calculator,
        provenance_tracker=prepared.provenance_tracker,
        variable_orderer=orderer,
        training_mask=partial(
            training_mask_at, prepared.window_calculator, is_panel=layout.is_panel
        ),
    )
    params = StageParams(
        covariate_strategy=imputer.covariate_strategy,
        fit_predict_order=imputer.fit_predict_order,
        estimator=imputer.estimator,
        impute_intermediate_frequencies=imputer.impute_intermediate_frequencies,
        aggregation_constraint=imputer.aggregation_constraint,
        impute_unobserved_entities=imputer.impute_unobserved_entities,
        target_column=prepared.target_column,
        target_as_covariate=imputer.target_as_covariate,
    )
    return StageRunner(run, layout, params, log=log), prepared.X_work


# Ajustement de toutes les étapes de la progression par l'exécuteur
def _fit_all_stages(runner: StageRunner, X_work: pd.DataFrame) -> List[StageOutcome]:
    """Run every stage of the progression, as phase 5 of the fit does.

    Args:
        runner: Runner of a fit.
        X_work: Working frame of the fit.

    Returns:
        The outcome of each stage, in progression order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with runner.run.aggregation_constraint.deferred_warnings():
            return [
                runner.fit_stage(X_work, stage_freq)
                for stage_freq in runner.layout.progression
            ]


# Entrées de la composition d'une variable à une étape
def _stage_inputs(runner: StageRunner, X_work: pd.DataFrame, stage_freq):
    """Build the stage frame and the frequencies by column of one stage, as 5a does.

    Args:
        runner: Runner of a fit.
        X_work: Working frame of the fit.
        stage_freq: Frequency of the stage.

    Returns:
        Tuple ``(stage_frame, freqs_by_column)``.
    """
    freqs_by_column = runner.layout.frequencies_by_column()
    mask = runner.stage_mask(X_work.index, stage_freq, kind='imputation')
    stage_frame = runner.run.materializer.stage_frame(
        grid_index=mask.index,
        stage_freq=stage_freq,
        detected_frequencies=freqs_by_column,
        source_data=X_work,
    )
    return stage_frame, freqs_by_column


# Ajustement silencieux de l'imputeur
def _fitted(imputer: HighFrequencyImputer, data: pd.DataFrame) -> HighFrequencyImputer:
    """Fit the imputer, swallowing the warnings the fit legitimately emits."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(data)


# Transformation silencieuse
def _transformed(imputer: HighFrequencyImputer, data: pd.DataFrame) -> pd.DataFrame:
    """Transform the data, swallowing the warnings the replay legitimately emits."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.transform(data)


# =============================================================================
# Grilles de la passe
# =============================================================================
class TestPassGrids:
    """The grids of the pass, read on the window calculator of the pass."""

    def test_grid_of_a_real_calculator_on_panel_f(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        stage = runner.layout.progression[-1]
        grid = runner.prediction_grid(X_work.index, stage, [FR, DE])
        assert set(grid.get_level_values('country')) == {'FR', 'DE'}
        assert len(grid) == 72


# =============================================================================
# Auxiliaires d'une étape
# =============================================================================
class TestStepHelpers:
    """Covariate selection, empty rows, estimator, scale factor and pairs of a step."""

    def test_covariate_empty_on_one_window_is_dropped(self, reference_timeseries):
        training = reference_timeseries
        prediction = reference_timeseries.assign(q1=np.nan)
        assert select_feature_columns(
            CovariateMaterializer(), ['m1', 'q1', 'a2'], training, prediction
        ) == ('m1', 'a2')
        assert select_feature_columns(
            CovariateMaterializer(), ['q1', 'm1'], prediction, training
        ) == ('m1',)

    def test_eligibility_is_read_per_entity(self, mixed_freq_panel_heterogeneous):
        frame = mixed_freq_panel_heterogeneous
        candidates = ['m1', 'climat_affaires']
        assert select_feature_columns(
            CovariateMaterializer(covariate_eligibility='any_entity'), candidates, frame, frame
        ) == ('m1', 'climat_affaires')
        assert select_feature_columns(
            CovariateMaterializer(covariate_eligibility='all_entities'), candidates, frame, frame
        ) == ('m1',)

    def test_rows_without_any_covariate_are_dropped(self):
        X = pd.DataFrame({'m1': [np.nan, 2.0, np.nan], 'q1': [np.nan, np.nan, 1.0]})
        y = pd.Series([1.0, 2.0, 3.0])
        X_kept, y_kept = drop_empty_training_rows(X, y)
        assert X_kept.index.tolist() == y_kept.index.tolist() == [1, 2]

    def test_no_covariate_leaves_the_pair_untouched(self):
        X, y = pd.DataFrame(index=range(3)), pd.Series([1.0, np.nan, 3.0])
        X_kept, y_kept = drop_empty_training_rows(X, y)
        assert X_kept is X and y_kept is y

    def test_estimator_resolution(self):
        spy = SpyEstimator(constant=2.0)
        default = SpyEstimator(constant=3.0)
        resolved = estimator_for(spy, 'a1')
        # Clone neuf : jamais l'objet du paramètre, mêmes hyperparamètres
        assert resolved is not spy and resolved.get_params() == spy.get_params()
        assert estimator_for({'a1': spy, '__default__': default}, 'm1').constant == 3.0
        assert estimator_for({'a1': spy}, 'm1') is None
        assert estimator_for(None, 'a1') is None

    def test_group_scale_factor_of_constant_mode(self):
        scaler = StageScaler(scale_features='constant', column_frequencies={'a1': 'Y', 'q1': 'Q'})
        grid = pd.DatetimeIndex([])
        assert group_scale_factor(scaler, 'a1', 'Y', 'M', grid) == 12.0
        assert group_scale_factor(scaler, 'q1', 'Q', 'M', grid) == 3.0

    def test_step_pairs_of_a_time_series_are_empty(self, reference_timeseries):
        imputer = _fitted(_imputer(), reference_timeseries)
        assert all(step_pairs(step) == [] for step in imputer.imputation_plan_)


class TestStageParams:
    """The parameters the stage execution reads."""

    def test_every_field_is_required_and_named(self):
        """No default: a parameter left out, or passed by position, fails at construction."""
        with pytest.raises(TypeError, match='missing 7 required keyword-only arguments'):
            StageParams(covariate_strategy='model')
        with pytest.raises(TypeError):
            StageParams('interpolate', 'frequency', None, False, 'sum', False, None, False)

    def test_the_imputer_passes_its_own_values(self, reference_timeseries):
        """Each field holds the very object of the imputer parameter of the same name."""
        imputer = _fitted(
            _imputer(
                covariate_strategy='model', impute_intermediate_frequencies='covariates_only',
                aggregation_constraint=None, impute_unobserved_entities=True,
            ),
            reference_timeseries,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prepared = prepare_work_frame(imputer, reference_timeseries, fitting=False)
        run = _RunContext.for_transform(
            imputer._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            transform_frequencies=prepared.new_frequencies,
            transform_misaligned=prepared.new_misaligned,
        )
        params = imputer._stage_runner(run).params
        imputer_params = imputer.get_params(deep=False)
        for field in fields(StageParams):
            expected = (
                imputer.target_column_ if field.name == 'target_column'
                else imputer_params[field.name]
            )
            assert getattr(params, field.name) is expected, field.name


# =============================================================================
# Composition d'une variable à une étape
# =============================================================================
class TestPrepareVariable:
    """The mutualized fit context of one (stage, variable), PANEL-F golden values."""

    @staticmethod
    def _prepare(runner, X_work, column, entities, stage_freq):
        stage_frame, freqs_by_column = _stage_inputs(runner, X_work, stage_freq)
        return runner.prepare_variable(
            column=column,
            entities=entities,
            stage_freq=stage_freq,
            X_work=X_work,
            stage_frame=stage_frame,
            freqs_by_column=freqs_by_column,
        )

    def test_stage_m_pools_51_rows_at_the_monthly_scale(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        prepared = self._prepare(runner, X_work, 'v', [DE, FR], runner.layout.progression[-1])

        # Trois blocs à leur propre fréquence, 3 + 12 + 36 lignes
        assert prepared.blocks == {FR: 'Y', DE: 'Q', IT: 'M'}
        assert len(prepared.y_train) == 51
        assert prepared.y_train.name == 'v'
        y = prepared.y_train
        assert y.loc['FR'].tolist() == [10.0, 11.0, 12.5]
        np.testing.assert_allclose(
            y.loc['DE'].to_numpy(),
            np.array([28, 30, 31, 31, 31, 33, 34, 34, 36, 37, 38, 39]) / 3,
        )
        assert y.loc['IT'].tolist() == [10.0] * 12 + [11.0] * 12 + [12.5] * 12
        # Covariables : celles de la grille de prédiction, voies décidées sur elle
        assert prepared.feature_cols == ('m1', 'q1')
        assert prepared.ways == dict(prepared.training.ways)
        assert prepared.X_train.index.equals(prepared.y_train.index)
        # Grille de prédiction restreinte aux entités des groupes
        assert set(prepared.pred_grid.get_level_values('country')) == {'FR', 'DE'}

    def test_stage_q_divides_by_4_1_and_a_third(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(
            _imputer(impute_intermediate_frequencies='covariates_only'),
            mixed_freq_panel_multifrequency,
        )
        prepared = self._prepare(runner, X_work, 'v', [FR], runner.layout.progression[0])
        y = prepared.y_train
        assert len(y) == 51
        assert y.loc['FR'].tolist() == [30.0, 33.0, 37.5]
        assert y.loc['DE'].tolist() == [28.0, 30.0, 31.0, 31.0, 31.0, 33.0,
                                        34.0, 34.0, 36.0, 37.0, 38.0, 39.0]
        np.testing.assert_allclose(
            y.loc['IT'].to_numpy(), [30.0] * 12 + [33.0] * 12 + [37.5] * 12
        )

    def test_no_register_is_written(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        before = runner.run.provenance_tracker.get_provenance_matrix()
        self._prepare(runner, X_work, 'v', [DE, FR], runner.layout.progression[-1])
        materializer = runner.run.materializer
        assert materializer.imputed_store == {}
        assert materializer.imputed_freq_store == {}
        assert materializer.origin_store == {}
        pd.testing.assert_frame_equal(
            runner.run.provenance_tracker.get_provenance_matrix(), before
        )

    def test_the_estimator_fits_the_composed_set(self, mixed_freq_panel_multifrequency):
        """The rank-1 variable is fitted on exactly the pair ``prepare_variable`` composes."""
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        stage = runner.layout.progression[-1]
        prepared = self._prepare(runner, X_work, 'q1', [FR, DE, IT], stage)

        outcome = runner.fit_stage(X_work, stage)

        model = next(step.model for step in outcome.steps if step.var_name == 'q1')
        pd.testing.assert_frame_equal(model.fit_X_, prepared.X_train, check_exact=True)
        pd.testing.assert_series_equal(model.fit_y_, prepared.y_train, check_exact=True)


# =============================================================================
# Fit d'une étape
# =============================================================================
class TestFitStage:
    """One fit per (stage, variable), one executed plan step per group."""

    def test_one_fit_shared_by_the_groups_of_a_variable(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        fits_before = SpyEstimator.n_fits
        (outcome,) = _fit_all_stages(runner, X_work)

        # q1 : un groupe ; v : deux groupes (DE trimestrielle, FR annuelle), un seul modèle
        assert [(step.var_name, step.source_frequency) for step in outcome.steps] == [
            ('q1', 'Q'), ('v', 'Q'), ('v', 'Y'),
        ]
        assert SpyEstimator.n_fits - fits_before == 2
        v_steps = [step for step in outcome.steps if step.var_name == 'v']
        assert v_steps[0].model is v_steps[1].model
        assert [step.var_key for step in v_steps] == [('v', 'Q'), ('v', 'Y')]
        assert [step.entities for step in v_steps] == [(DE,), (FR,)]
        assert {step.n_training_rows for step in v_steps} == {51}
        # Facteur partagé : le même objet porte les deux champs
        assert all(step.scale_factor is step.fit_scale_factor for step in outcome.steps)
        assert all(step.n_written > 0 for step in outcome.steps)

    def test_no_order_outside_the_model_strategy(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)
        (outcome,) = _fit_all_stages(runner, X_work)
        assert outcome.order is None and outcome.cv_scores == {}

    def test_frequency_order_under_the_model_strategy(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(
            _imputer(covariate_strategy='model'), mixed_freq_panel_multifrequency
        )
        (outcome,) = _fit_all_stages(runner, X_work)
        # v, annuelle chez FR, précède q1, trimestrielle
        assert outcome.order == ['v', 'q1']
        assert [step.var_name for step in outcome.steps] == ['v', 'v', 'q1']
        assert outcome.cv_scores == {}

    def test_cv_order_reports_its_scores(self, mixed_freq_panel_multifrequency):
        runner, X_work = _fit_runner(
            _imputer(covariate_strategy='model', fit_predict_order='cv', cv=3),
            mixed_freq_panel_multifrequency,
        )
        (outcome,) = _fit_all_stages(runner, X_work)
        assert sorted(outcome.order) == ['q1', 'v']
        assert set(outcome.cv_scores) == {'q1', 'v'}

    def test_stage_without_imputable_variable(self, reference_timeseries):
        messages = []
        runner, X_work = _fit_runner(_imputer(), reference_timeseries[['m1']], log=messages.append)
        outcome = runner.fit_stage(X_work, 'M')
        assert outcome == StageOutcome(stage_label='M')
        assert messages == ['[fit] stage M: no imputable variable']

    def test_failing_fit_falls_back_on_interpolation(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(estimator=FailingEstimator()), reference_timeseries)
        (outcome,) = _fit_all_stages(runner, X_work)
        assert all(step.is_fallback for step in outcome.steps)
        assert all(step.model == INTERPOLATE_FALLBACK for step in outcome.steps)
        assert all(step.feature_cols == () and step.n_written > 0 for step in outcome.steps)
        assert len(runner.run.pending_warnings.degraded_steps) == 3
        # Lignes d'ancre comprises : la valeur réexprimée ne porte pas l'observation
        provenance = runner.run.provenance_tracker.get_provenance_matrix()
        assert set(provenance['a1'].astype(str)) == {'interpolated'}

    def test_failing_prediction_degrades_the_step(self, reference_timeseries):
        runner, X_work = _fit_runner(
            _imputer(estimator=_PredictFailingEstimator()), reference_timeseries
        )
        (outcome,) = _fit_all_stages(runner, X_work)
        # Ajusté, puis dégradé en repli à l'exécution : le plan dit ce qui a été fait
        for step in outcome.steps:
            assert step.is_fallback and step.model == INTERPOLATE_FALLBACK
            assert step.feature_cols == () and step.materialization == {}
            assert (step.covariate_taint, step.target_taint) == ('none', 'none')
            assert step.n_written > 0
        assert all(
            'prediction failed (RuntimeError: deliberate predict failure)' in message
            for message in runner.run.pending_warnings.degraded_steps
        )

    def test_unanchored_group_shares_the_model_and_is_not_rescaled(
        self, mixed_freq_panel_multifrequency
    ):
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['IT', 'v'] = np.nan
        runner, X_work = _fit_runner(_imputer(impute_unobserved_entities=True), data)
        (outcome,) = _fit_all_stages(runner, X_work)

        v_steps = [step for step in outcome.steps if step.var_name == 'v']
        assert [(step.source_frequency, step.unanchored) for step in v_steps] == [
            ('Q', False), ('Y', False), (None, True),
        ]
        model = v_steps[0].model
        assert all(step.model is model for step in v_steps)
        assert (v_steps[2].scale_factor, v_steps[2].fit_scale_factor) == (1.0, 1.0)
        # IT n'apporte aucune ligne (15 = 3 FR + 12 DE)
        assert v_steps[0].training_blocks == {FR: 'Y', DE: 'Q'}
        assert {step.n_training_rows for step in v_steps} == {15}

        # Prédiction libre chez IT, recalage aux totaux annuels chez FR
        values = runner.run.materializer.imputed_store['v']
        assert values.loc['IT'].tolist() == [model.mean_] * 36
        assert values.loc['FR'].loc['2021'].tolist() == [10.0] * 12
        provenance = runner.run.provenance_tracker.get_provenance_matrix()
        assert set(provenance.loc['IT', 'v'].astype(str)) == {'model_unanchored'}
        assert runner.run.unanchored_written == {('IT', 'v')}

    def test_unanchored_prediction_failure_leaves_the_pair_unmarked(
        self, mixed_freq_panel_multifrequency
    ):
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['IT', 'v'] = np.nan
        runner, X_work = _fit_runner(
            _imputer(estimator=_PredictFailingEstimator(), impute_unobserved_entities=True), data
        )
        (outcome,) = _fit_all_stages(runner, X_work)

        unanchored = [step for step in outcome.steps if step.unanchored]
        assert len(unanchored) == 1
        assert not unanchored[0].is_fallback and unanchored[0].n_written == 0
        assert runner.run.pending_warnings.unimputed_pairs == [('IT', 'v')]
        assert runner.run.unanchored_written == set()
        provenance = runner.run.provenance_tracker.get_provenance_matrix()
        assert provenance.loc['IT', 'v'].isna().all()

    def test_column_without_estimator_falls_back(self, reference_timeseries):
        runner, X_work = _fit_runner(
            _imputer(estimator={'a1': SpyEstimator()}), reference_timeseries
        )
        (outcome,) = _fit_all_stages(runner, X_work)
        fallbacks = {step.var_name: step.is_fallback for step in outcome.steps}
        assert fallbacks == {'a1': False, 'a2': True, 'q1': True}
        assert sorted(runner.run.pending_warnings.degraded_steps) == [
            "'a2' at stage M: no estimator covers this column, interpolation fallback",
            "'q1' at stage M: no estimator covers this column, interpolation fallback",
        ]

    def test_no_estimator_falls_back_silently(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(estimator=None), reference_timeseries)
        (outcome,) = _fit_all_stages(runner, X_work)
        assert all(step.is_fallback for step in outcome.steps)
        # L'avertissement global de la phase 0 suffit : aucun message par étape
        assert runner.run.pending_warnings.degraded_steps == []

    def test_empty_prediction_window_selects_no_covariate(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(), reference_timeseries)
        runner.run.window_calculator = _FixedMaskCalculator(pd.Series(False, index=X_work.index))

        (outcome,) = _fit_all_stages(runner, X_work)

        # Lignes d'entraînement présentes, aucune covariable disponible à la prédiction
        assert {step.var_name: step.n_training_rows for step in outcome.steps} == {
            'q1': 12, 'a1': 3, 'a2': 3,
        }
        assert all(step.is_fallback and step.feature_cols == () for step in outcome.steps)
        assert all(step.n_written == 0 for step in outcome.steps)
        assert runner.run.pending_warnings.degraded_steps == []

    def test_empty_windows_fit_on_nothing_and_announce_nothing(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(estimator=FailingEstimator()), reference_timeseries)
        # Fenêtre d'entraînement vide (jeu mutualisé sans ligne) et fenêtre
        # d'imputation vide (grille de prédiction sans ligne)
        runner.run.training_set_builder.training_mask = (
            lambda frequencies: pd.Series(False, index=X_work.index)
        )
        runner.run.window_calculator = _FixedMaskCalculator(pd.Series(False, index=X_work.index))

        (outcome,) = _fit_all_stages(runner, X_work)

        assert all(step.n_training_rows == 0 for step in outcome.steps)
        assert all(step.is_fallback and step.n_written == 0 for step in outcome.steps)
        # Rien à écrire : le repli n'est pas annoncé, le calculateur l'a déjà dit
        assert runner.run.pending_warnings.degraded_steps == []

    def test_unanchored_entity_outside_any_window_is_predicted(
        self, mixed_freq_panel_multifrequency
    ):
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['IT', 'v'] = np.nan
        runner, X_work = _fit_runner(_imputer(impute_unobserved_entities=True), data)
        # Fenêtre d'imputation vide chez IT seulement
        in_window = X_work.index.get_level_values('country') != 'IT'
        runner.run.window_calculator = _FixedMaskCalculator(
            pd.Series(in_window, index=X_work.index)
        )

        (outcome,) = _fit_all_stages(runner, X_work)

        unanchored = [step for step in outcome.steps if step.unanchored]
        assert [step.n_written for step in unanchored] == [36]

    def test_nan_predictions(self, mixed_freq_panel_multifrequency):
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['IT', 'v'] = np.nan
        runner, X_work = _fit_runner(
            _imputer(estimator=_NaNPredictingEstimator(), impute_unobserved_entities=True), data
        )
        (outcome,) = _fit_all_stages(runner, X_work)

        # Ancrées : dégradées en repli ; sans ancre : couple signalé, jamais un repli
        degraded = runner.run.pending_warnings.degraded_steps
        assert all('the model predicted only NaN' in message for message in degraded)
        assert len(degraded) == len([step for step in outcome.steps if not step.unanchored])
        assert runner.run.pending_warnings.unimputed_pairs == [('IT', 'v')]

    def test_scattered_nan_predictions_stay_unwritten(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(estimator=_FirstNaNEstimator()), reference_timeseries)
        (outcome,) = _fit_all_stages(runner, X_work)
        assert all(not step.is_fallback and step.n_written == 35 for step in outcome.steps)
        assert all(
            '1 of 36 prediction(s) are NaN and stay unimputed' in message
            for message in runner.run.pending_warnings.degraded_steps
        )

    def test_replay_context_is_refused(self, reference_timeseries):
        runner, X_work = _fit_runner(_imputer(), reference_timeseries)
        components = FittedComponents.build(
            covariate_strategy='interpolate', covariate_fallback='interpolate',
            covariate_eligibility='any_entity', interpolation_method='linear',
            interpolation_anchor=None, aggregation_constraint='sum',
            scale_features='constant', column_frequencies=runner.layout.frequencies_by_column(),
        )
        replay_run = _RunContext.for_transform(
            components,
            window_calculator=ImputationWindowCalculator(),
            provenance_tracker=ImputationProvenanceTracker(),
            transform_frequencies={},
            transform_misaligned=(),
        )
        with pytest.raises(RuntimeError, match='needs the execution context of a fit'):
            StageRunner(replay_run, runner.layout, runner.params).fit_stage(X_work, 'M')

    @pytest.mark.parametrize(
        'dataset, overrides',
        [
            ('mixed_freq_panel_multifrequency', {}),
            ('mixed_freq_panel_multifrequency',
             {'covariate_strategy': 'model', 'impute_intermediate_frequencies': True}),
            pytest.param('panel_reference_full', {}, marks=pytest.mark.slow),
            pytest.param(
                'panel_reference_full',
                {'covariate_strategy': 'model', 'impute_intermediate_frequencies': True},
                marks=pytest.mark.slow,
            ),
            pytest.param(
                'panel_reference_full',
                {'covariate_strategy': 'model', 'fit_predict_order': 'cv', 'cv': 3,
                 'impute_intermediate_frequencies': 'covariates_only',
                 'aggregation_constraint': None},
                marks=pytest.mark.slow,
            ),
        ],
        ids=[
            'panel-f-default', 'panel-f-model-axis2',
            'panel-x-default', 'panel-x-model-axis2', 'panel-x-model-cv-covariates-only',
        ],
    )
    def test_stages_reproduce_the_plan_of_the_imputer(self, dataset, overrides, request):
        """The runner wired from the public blocks yields the imputer's plan, fits included."""
        data = request.getfixturevalue(dataset)
        runner, X_work = _fit_runner(_imputer(**overrides), data)
        plan = ImputationPlan()
        for outcome in _fit_all_stages(runner, X_work):
            for step in outcome.steps:
                plan = append_step(plan, step)
        imputer = _fitted(_imputer(**overrides), data)

        pd.testing.assert_frame_equal(
            plan.to_diagnostic_frame(), imputer.imputation_plan_.to_diagnostic_frame()
        )
        for ours, theirs in zip(plan, imputer.imputation_plan_):
            if hasattr(theirs.model, 'fit_X_'):
                pd.testing.assert_frame_equal(ours.model.fit_X_, theirs.model.fit_X_, check_exact=True)
                pd.testing.assert_series_equal(ours.model.fit_y_, theirs.model.fit_y_, check_exact=True)


# =============================================================================
# Exécution d'une étape
# =============================================================================
class TestExecuteStep:
    """The single execution of a frozen step."""

    def test_writes_values_provenance_and_registers(self, reference_timeseries):
        imputer = _fitted(_imputer(), reference_timeseries)
        step = next(step for step in imputer.imputation_plan_ if step.var_name == 'q1')
        runner, X_work = _fit_runner(_imputer(), reference_timeseries)

        degraded, n_written = runner.execute_step(step, X_work=X_work, stage_freq='M')

        assert (degraded, n_written) == (False, step.n_written) == (False, 36)
        materializer = runner.run.materializer
        assert materializer.imputed_store['q1'].notna().sum() == 36
        assert set(materializer.origin_store['q1'].astype(str)) == {'model'}
        provenance = runner.run.provenance_tracker.get_provenance_matrix()
        assert set(provenance['q1'].astype(str)) == {step.emitted_provenance.value}

    def test_empty_grid_writes_nothing(self, mixed_freq_panel_multifrequency):
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        step = replace(imputer.imputation_plan_.steps[0], entities=(('ES',),))
        runner, X_work = _fit_runner(_imputer(), mixed_freq_panel_multifrequency)

        result = runner.execute_step(
            step, X_work=X_work, stage_freq=runner.layout.progression[-1]
        )

        assert result == (False, 0)
        assert runner.run.materializer.imputed_store == {}

    def test_fallback_producing_nothing_writes_nothing(self, reference_timeseries):
        imputer = _fitted(_imputer(estimator=FailingEstimator()), reference_timeseries)
        step = next(step for step in imputer.imputation_plan_ if step.var_name == 'a1')
        runner, X_work = _fit_runner(_imputer(), reference_timeseries)

        # Colonne vide sur les données de l'appel : l'interpolation ne produit rien
        result = runner.execute_step(step, X_work=X_work.assign(a1=np.nan), stage_freq='M')

        assert result == (False, 0)
        assert 'a1' not in runner.run.materializer.imputed_store

    def test_replay_step_delegates_to_execute_step(self, reference_timeseries, monkeypatch):
        imputer = _fitted(_imputer(), reference_timeseries)
        step = imputer.imputation_plan_.steps[0]
        runner, X_work = _fit_runner(_imputer(), reference_timeseries)
        calls = []

        def _spy(self, step, *, X_work, stage_freq):
            calls.append((self, step, X_work, stage_freq))
            return True, 12

        monkeypatch.setattr(StageRunner, 'execute_step', _spy)
        assert runner.replay_step(step, X_work=X_work, stage_freq='M') is None
        assert calls == [(runner, step, X_work, 'M')]


# =============================================================================
# Un seul chemin d'exécution pour le fit et le rejeu
# =============================================================================
class TestSingleExecutionPath:
    """Fit and replay execute every step through ``StageRunner.execute_step``."""

    @staticmethod
    def _record_executions(monkeypatch) -> list:
        """Wrap ``execute_step`` and ``replay_step`` so each call is recorded, then delegated."""
        calls = []
        execute, replay = StageRunner.execute_step, StageRunner.replay_step

        def _execute(self, step, **kwargs):
            calls.append(('execute', step.var_name))
            return execute(self, step, **kwargs)

        def _replay(self, step, **kwargs):
            calls.append(('replay', step.var_name))
            return replay(self, step, **kwargs)

        monkeypatch.setattr(StageRunner, 'execute_step', _execute)
        monkeypatch.setattr(StageRunner, 'replay_step', _replay)
        return calls

    @pytest.mark.parametrize(
        'dataset',
        ['mixed_freq_panel_multifrequency',
         pytest.param('panel_reference_full', marks=pytest.mark.slow)],
        ids=['panel-f', 'panel-x'],
    )
    def test_fit_and_replay_call_the_same_execution(self, dataset, request, monkeypatch):
        data = request.getfixturevalue(dataset)
        calls = self._record_executions(monkeypatch)
        imputer = _fitted(
            _imputer(covariate_strategy='model', impute_intermediate_frequencies=True), data
        )
        plan_columns = [step.var_name for step in imputer.imputation_plan_]
        assert calls == [('execute', column) for column in plan_columns]

        calls.clear()
        _transformed(imputer, data)
        # Chaque étape rejouée passe par replay_step, qui délègue à execute_step
        assert calls == [
            call for column in plan_columns
            for call in (('replay', column), ('execute', column))
        ]

    def test_neutralized_execution_imputes_nothing(self, panel_reference_full, monkeypatch):
        monkeypatch.setattr(StageRunner, 'execute_step', lambda self, step, **kwargs: (False, 0))
        imputer = _fitted(_imputer(covariate_strategy='model'), panel_reference_full)

        assert len(imputer.imputation_plan_) > 0
        assert all(step.n_written == 0 for step in imputer.imputation_plan_)
        fit_marks = set(imputer.imputation_provenance_.stack().astype(str))
        assert fit_marks.isdisjoint(_STEP_PROVENANCES)

        _transformed(imputer, panel_reference_full)
        transform_marks = set(imputer.imputation_provenance_.stack().astype(str))
        assert transform_marks.isdisjoint(_STEP_PROVENANCES)

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_replay_on_the_fit_data_predicts_on_the_same_frames(
        self, strategy, mixed_freq_panel_multifrequency
    ):
        imputer = _fitted(
            _imputer(covariate_strategy=strategy), mixed_freq_panel_multifrequency
        )
        models = list({id(step.model): step.model for step in imputer.imputation_plan_}.values())
        at_fit = {id(model): list(model.predict_X_) for model in models}

        _transformed(imputer, mixed_freq_panel_multifrequency)

        for model in models:
            frames = at_fit[id(model)]
            assert len(model.predict_X_) == 2 * len(frames)
            for seen_at_fit, seen_at_replay in zip(frames, model.predict_X_[len(frames):]):
                pd.testing.assert_frame_equal(seen_at_replay, seen_at_fit, check_exact=True)
