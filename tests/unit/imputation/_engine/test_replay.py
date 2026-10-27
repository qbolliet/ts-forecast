"""Tests of the plan replay of ``tsforecast.imputation._engine.replay``.

Scenarios: the frequency check of the transform (same data, missing column, entirely empty column
of the fit, extra column, diverging frequency, pairs unknown to the fit, pairs off their calendar
bounds); the warning on the rows outside the recomputed window (time series, panel, entity without
window, unusable calculator); the guard of the prediction pass; the entities born at transform and
their stage binding; the steps one stage replays (parameter off, unanchored and anchored derived
steps, fit-unanchored pair now observed, misaligned pair); and the replay of the whole plan on
PANEL-F (frames identical to ``transform``, one ``replay_step`` per step, skipped columns, notes of
the aggregated warning, logging).
"""
# Modules de base
import warnings
from types import SimpleNamespace
from typing import List, Tuple

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.imputation import HighFrequencyImputer
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.imputation._engine.frequency_layout import stage_frequency_label
from tsforecast.imputation._engine.preparation import PreparedData, prepare_work_frame
from tsforecast.imputation._engine.replay import (
    check_prediction_pass_replayable,
    check_transform_frequencies,
    extend_stage_binding,
    new_entities,
    replay_plan,
    replay_steps,
    warn_rows_outside_window,
)
from tsforecast.imputation._engine.run_context import _RunContext
from tsforecast.imputation._engine.stage_runner import StageRunner
from tsforecast.imputation._engine.warnings import WarningAccumulator

# Clés d'entité des jeux de référence
FR, DE, IT, ES = ('FR',), ('DE',), ('IT',), ('ES',)
# Liaison de l'étape mensuelle de PANEL-F
_MONTHLY = {DE: 'M', FR: 'M', IT: 'M'}


# =============================================================================
# Constructeurs
# =============================================================================
# Imputeur non ajusté, estimateur linéaire par défaut
def _imputer(**overrides) -> HighFrequencyImputer:
    """Build an unfitted imputer targeting the monthly frequency.

    Args:
        **overrides: Constructor parameters replacing the defaults
            ``target_frequency='M'`` and ``estimator=LinearRegression()``.

    Returns:
        The imputer.
    """
    params = dict(target_frequency='M', estimator=LinearRegression())
    params.update(overrides)
    return HighFrequencyImputer(**params)


# Ajustement silencieux
def _fitted(imputer: HighFrequencyImputer, data: pd.DataFrame, y=None) -> HighFrequencyImputer:
    """Fit the imputer, swallowing the warnings the fit legitimately emits."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(data, y)


# Panel augmenté d'une entité copiée
def _with_entity(
    data: pd.DataFrame,
    source: str = 'IT',
    name: str = 'ES',
    clear: Tuple[str, ...] = ('v',),
) -> pd.DataFrame:
    """Return a panel extended by a renamed copy of one entity.

    Args:
        data: Panel with an ``(entity, date)`` MultiIndex.
        source: Entity copied.
        name: Name of the new entity.
        clear: Columns emptied on the copy.

    Returns:
        The extended panel, sorted.
    """
    added = data.xs(source, level=0, drop_level=False).copy()
    added.index = pd.MultiIndex.from_arrays(
        [[name] * len(added), added.index.get_level_values(-1)], names=data.index.names
    )
    for column in clear:
        added[column] = np.nan
    return pd.concat([data, added]).sort_index()


# Exécuteur d'un rejeu câblé comme l'imputeur le câble
def _replay_runner(
    imputer: HighFrequencyImputer,
    data: pd.DataFrame,
    *,
    log=None,
) -> Tuple[StageRunner, PreparedData]:
    """Build the stage runner of a replay on ``data``, before any step runs.

    Args:
        imputer: Fitted imputer.
        data: Data of the transform.
        log: Logging callback of the runner, None to keep the imputer's.

    Returns:
        Tuple ``(runner, prepared)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = prepare_work_frame(imputer, data, fitting=False)
    run = _RunContext.for_transform(
        imputer._fitted_components_,
        window_calculator=prepared.window_calculator,
        provenance_tracker=prepared.provenance_tracker,
        transform_frequencies=prepared.new_frequencies,
        transform_misaligned=prepared.new_misaligned,
    )
    runner = imputer._stage_runner(run)
    if log is not None:
        runner.log = log
    return runner, prepared


# Étapes rejouées à l'étape mensuelle de PANEL-F, entités nouvelles liées
def _monthly_replay(imputer: HighFrequencyImputer, data: pd.DataFrame) -> list:
    """Compose the steps of the monthly stage of PANEL-F for a transform on ``data``.

    Args:
        imputer: Imputer fitted on PANEL-F (or a variant of it).
        data: Data of the transform.

    Returns:
        ``(var_name, source_frequency, entities, unanchored)`` of each step,
        in execution order.
    """
    runner, prepared = _replay_runner(imputer, data)
    born = new_entities(prepared.X_work.index, imputer.entities_, is_panel=True)
    binding = {**_MONTHLY, **{entity: 'M' for entity in born}}
    steps = replay_steps(
        runner, imputer.imputation_plan_.by_stage()['M'], born, binding, prepared.X_work
    )
    return [
        (step.var_name, step.source_frequency, step.entities, step.unanchored)
        for step in steps
    ]


# =============================================================================
# Contrôle des fréquences au transform
# =============================================================================
class TestCheckTransformFrequencies:
    """The data of the transform are checked pair by pair against the fit."""

    def test_the_fit_data_carry_no_new_pair(self, mixed_freq_panel_multifrequency):
        """On the data of the fit, nothing is new and nothing warns."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = check_transform_frequencies(
                imputer._frequency_layout_, mixed_freq_panel_multifrequency
            )
        assert result == ({}, ())

    def test_missing_column_raises_naming_it(self, reference_timeseries):
        """A column of the fit absent from the transform is refused by name."""
        imputer = _fitted(_imputer(), reference_timeseries)
        with pytest.raises(ValueError, match=r"missing 2 column\(s\) seen at fit time: \['a1', 'q1'\]"):
            check_transform_frequencies(
                imputer._frequency_layout_, reference_timeseries[['m1', 'a2']]
            )

    def test_optional_column_may_be_absent(self, reference_timeseries):
        """A column declared optional — the target of the prediction pass — may be missing."""
        imputer = _fitted(_imputer(), reference_timeseries)
        result = check_transform_frequencies(
            imputer._frequency_layout_, reference_timeseries.drop(columns='a2'),
            optional_columns=('a2',),
        )
        assert result == ({}, ())

    def test_entirely_empty_column_of_the_fit_is_still_required(self, reference_timeseries):
        """A column empty at fit time — undetectable — is part of the frozen plan."""
        data = reference_timeseries.assign(a2=np.nan)
        imputer = _fitted(_imputer(), data)
        assert 'a2' in imputer._frequency_layout_.undetected
        with pytest.raises(ValueError, match=r"\['a2'\]"):
            check_transform_frequencies(imputer._frequency_layout_, data.drop(columns='a2'))

    def test_extra_column_is_ignored_silently(self, reference_timeseries):
        """A column the fit never saw belongs to no plan: no error, no warning, no pair."""
        imputer = _fitted(_imputer(), reference_timeseries)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = check_transform_frequencies(
                imputer._frequency_layout_, reference_timeseries.assign(extra=1.0)
            )
        assert result == ({}, ())

    def test_diverging_frequency_warns_once_and_keeps_the_fit(self, reference_timeseries):
        """A pair observed at another frequency warns once; the fit frequency is kept."""
        imputer = _fitted(_imputer(), reference_timeseries)
        diverging = reference_timeseries.copy()
        diverging['q1'] = np.nan
        diverging.loc[diverging.index.month == 12, 'q1'] = [30.0, 90.0, 150.0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = check_transform_frequencies(imputer._frequency_layout_, diverging)

        messages = [str(item.message) for item in caught]
        assert len(messages) == 1
        assert messages[0].startswith(
            "1 (entity, column) pair(s) carry a different frequency at transform "
            "time; the fit frequencies are kept: 'q1': fit=Q, transform="
        )
        # Couple connu du fit : jamais "nouveau"
        assert result == ({}, ())
        assert imputer._frequency_layout_.detected_frequencies['q1'] == 'Q'

    def test_entity_born_at_transform_gets_its_detected_frequencies(
        self, mixed_freq_panel_multifrequency
    ):
        """The pairs of a new entity carry the frequency detected on the transform data."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        result = check_transform_frequencies(imputer._frequency_layout_, data)
        assert result == ({('ES', 'm1'): 'M', ('ES', 'q1'): 'Q', ('ES', 'v'): 'Q'}, ())

    def test_column_left_empty_at_fit_is_a_new_pair(self, mixed_freq_panel_multifrequency):
        """A pair a known entity left empty at fit time is new when the transform observes it."""
        blank = mixed_freq_panel_multifrequency.copy()
        blank.loc['DE', 'v'] = np.nan
        imputer = _fitted(_imputer(), blank)
        result = check_transform_frequencies(
            imputer._frequency_layout_, mixed_freq_panel_multifrequency
        )
        assert result == ({('DE', 'v'): 'Q'}, ())

    def test_new_pair_off_its_calendar_bounds_is_set_apart(self, mixed_freq_panel_multifrequency):
        """A new pair dated off the bounds of its own frequency is named, not imputed."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency, source='DE')
        values = np.full(36, np.nan)
        # Trimestrielle datée un mois après la fin de chaque trimestre
        values[1::3] = 5.0
        data.loc['ES', 'v'] = values
        frequencies, misaligned = check_transform_frequencies(imputer._frequency_layout_, data)
        assert misaligned == (('ES', 'v'),)
        assert ('ES', 'v') not in frequencies


# =============================================================================
# Lignes hors fenêtre
# =============================================================================
class TestWarnRowsOutsideWindow:
    """One aggregated warning names the rows the recomputed window leaves out."""

    @staticmethod
    def _messages(calculator, data, *, is_panel) -> List[str]:
        """Return the messages ``warn_rows_outside_window`` emits."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warn_rows_outside_window(calculator, data, is_panel=is_panel)
        return [str(item.message) for item in caught]

    def test_rows_inside_the_window_are_silent(self, reference_timeseries):
        """A window covering every row produces no message."""
        data = reference_timeseries[['m1']]
        calculator = ImputationWindowCalculator().fit(data)
        assert self._messages(calculator, data, is_panel=False) == []

    def test_time_series_counts_the_rows_left_out(self):
        """The rows before the common coverage are counted, without entity names."""
        dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        data = pd.DataFrame(
            {'m1': np.arange(6.0), 'm2': [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]}, index=dates
        )
        calculator = ImputationWindowCalculator().fit(data)
        assert self._messages(calculator, data, is_panel=False) == [
            "2 row(s) of the data being transformed fall outside the imputation "
            "window recomputed on them. These rows keep their input values: "
            "nothing is ever blanked."
        ]

    def test_panel_names_the_entities_and_skips_those_without_window(self):
        """Only an entity that has a window is counted; it is named in the message."""
        dates = pd.date_range('2021-01-31', periods=4, freq='ME')
        index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        data = pd.DataFrame({
            'm1': np.arange(8.0),
            # FR : couverture commune à partir du deuxième mois ; DE : jamais
            'm2': [np.nan, 1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan],
        }, index=index)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            calculator = ImputationWindowCalculator().fit(data)
        assert self._messages(calculator, data, is_panel=True) == [
            "1 row(s) of the data being transformed fall outside the imputation "
            "window recomputed on them, for entities (('FR',),). These rows keep "
            "their input values: nothing is ever blanked."
        ]

    def test_unusable_calculator_is_silent(self, reference_timeseries):
        """An unfitted calculator — the window could not be computed — warns nothing more."""
        assert self._messages(
            ImputationWindowCalculator(), reference_timeseries, is_panel=False
        ) == []


# =============================================================================
# Garde de la passe de prédiction
# =============================================================================
class TestCheckPredictionPassReplayable:
    """No model of a column of X may read the target in the prediction pass."""

    @staticmethod
    def _step(var_name='q1', feature_cols=('m1',), is_fallback=False) -> SimpleNamespace:
        """Return a stand-in of a plan step: the guard reads three of its fields only."""
        return SimpleNamespace(
            var_name=var_name, feature_cols=feature_cols, is_fallback=is_fallback
        )

    def test_plan_without_target_covariate_passes(self):
        """A plan whose covariates never include the target is replayable."""
        assert check_prediction_pass_replayable([self._step()], 'y') is None

    def test_model_reading_the_target_is_refused(self):
        """A step of another column reading the target raises, naming every such column once."""
        steps = [
            self._step('q1', ('m1', 'y')),
            self._step('q1', ('y',)),
            self._step('a1', ('y',)),
        ]
        with pytest.raises(ValueError, match=r"cannot impute \['a1', 'q1'\]"):
            check_prediction_pass_replayable(steps, 'y')

    def test_fallback_and_target_steps_are_exempt(self):
        """A fallback step reads no covariate, and the target's own steps are not replayed."""
        steps = [
            self._step('q1', ('y',), is_fallback=True),
            self._step('y', ('y', 'm1')),
        ]
        assert check_prediction_pass_replayable(steps, 'y') is None

    def test_imputer_fitted_with_the_target_as_covariate(self, reference_timeseries):
        """The guard refuses the plan of an imputer fitted with ``target_as_covariate=True``."""
        y = reference_timeseries['m1'].rename('y') * 2.0
        imputer = _fitted(
            _imputer(covariate_strategy='model', target_as_covariate=True),
            reference_timeseries, y,
        )
        with pytest.raises(ValueError, match='target_as_covariate=True'):
            check_prediction_pass_replayable(imputer.imputation_plan_, imputer.target_column_)


# =============================================================================
# Entités nées au transform
# =============================================================================
class TestNewEntities:
    """The entities of the transform the fit never saw, in order of appearance."""

    def test_time_series_has_none(self, reference_timeseries):
        """A time series has no entity, hence no new one."""
        assert new_entities(reference_timeseries.index, None, is_panel=False) == ()

    def test_panel_lists_the_new_ones_in_order(self, mixed_freq_panel_multifrequency):
        """The unknown entities come in their order of appearance, normalized."""
        data = _with_entity(_with_entity(mixed_freq_panel_multifrequency, name='AT'), name='BE')
        assert new_entities(data.index, [FR, DE, IT], is_panel=True) == (('AT',), ('BE',))

    def test_scalar_keys_of_the_fit_are_normalized(self, mixed_freq_panel_multifrequency):
        """``'FR'`` and ``('FR',)`` name one entity."""
        assert new_entities(
            mixed_freq_panel_multifrequency.index, ['FR', 'DE', 'IT'], is_panel=True
        ) == ()

    def test_panel_without_known_entities_has_none(self, mixed_freq_panel_multifrequency):
        """Without entities recorded at fit time, nothing can be new."""
        assert new_entities(mixed_freq_panel_multifrequency.index, [], is_panel=True) == ()


class TestExtendStageBinding:
    """A new entity joins a stage only when its entities agree on one frequency."""

    def test_scalar_stage_is_returned_as_is(self):
        """The binding of a time series is a scalar: nothing to extend."""
        pending = WarningAccumulator('transform')
        assert extend_stage_binding(pending, 'M', [ES], 'M') == 'M'
        assert pending.messages() == []

    def test_no_new_entity_returns_the_same_binding(self):
        """Without new entities the binding object itself comes back."""
        pending = WarningAccumulator('transform')
        assert extend_stage_binding(pending, _MONTHLY, (), 'M') is _MONTHLY

    def test_unanimous_stage_binds_the_new_entities(self):
        """The shared frequency, in base form, is given to every new entity."""
        pending = WarningAccumulator('transform')
        extended = extend_stage_binding(pending, {FR: 'ME', DE: 'M'}, [ES, ('AT',)], 'M')
        assert extended == {FR: 'ME', DE: 'M', ES: 'M', ('AT',): 'M'}
        assert pending.messages() == []

    def test_disagreeing_stage_skips_them_and_notes_it(self):
        """Entities that disagree leave nothing to depart them: a note, and no binding."""
        pending = WarningAccumulator('transform')
        binding = {FR: 'M', DE: 'Q'}
        assert extend_stage_binding(pending, binding, [ES], 'M|Q') is binding
        assert len(pending.messages()) == 1
        assert (
            "stage M|Q: its entities disagree on the stage frequency, so the "
            "entities born at transform time (('ES',),) cannot be bound to it "
            "and are skipped"
        ) in pending.messages()[0]


# =============================================================================
# Étapes rejouées
# =============================================================================
class TestReplaySteps:
    """The steps one stage replays, derived steps included."""

    def test_parameter_off_replays_the_plan_alone(self, mixed_freq_panel_multifrequency):
        """Under ``impute_unobserved_entities=False`` the plan steps come back as they are."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency)
        runner, prepared = _replay_runner(imputer, data)
        steps = imputer.imputation_plan_.by_stage()['M']
        composed = replay_steps(runner, steps, [ES], {**_MONTHLY, ES: 'M'}, prepared.X_work)
        assert composed == list(steps)
        assert all(left is right for left, right in zip(composed, steps))

    def test_new_entity_never_observing_a_column_is_unanchored(
        self, mixed_freq_panel_multifrequency
    ):
        """A column the new entity never observes is served by an unanchored derived step."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        composed = _monthly_replay(imputer, _with_entity(mixed_freq_panel_multifrequency))
        assert composed == [
            ('q1', 'Q', (DE, FR, IT), False),
            ('q1', 'Q', (ES,), False),
            ('v', 'Q', (DE,), False),
            ('v', 'Y', (FR,), False),
            ('v', None, (ES,), True),
        ]

    def test_new_entity_observing_a_coarser_column_is_anchored(
        self, mixed_freq_panel_multifrequency
    ):
        """An observed column coarser than the stage gets an anchored step at its own frequency."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        composed = _monthly_replay(
            imputer, _with_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        )
        assert composed[-1] == ('v', 'Q', (ES,), False)

    def test_new_entity_observing_at_the_stage_frequency_is_left_as_given(
        self, mixed_freq_panel_multifrequency
    ):
        """A column already at the stage frequency is never imputed, as at fit time."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        composed = _monthly_replay(
            imputer, _with_entity(mixed_freq_panel_multifrequency, source='IT', clear=())
        )
        assert [step for step in composed if ES in step[2] and step[0] == 'v'] == []

    def test_derived_steps_share_the_frozen_decisions(self, mixed_freq_panel_multifrequency):
        """A derived step keeps model, covariates and ways; only scope, scale and anchoring change."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        runner, prepared = _replay_runner(imputer, _with_entity(mixed_freq_panel_multifrequency))
        steps = imputer.imputation_plan_.by_stage()['M']
        composed = replay_steps(runner, steps, [ES], {**_MONTHLY, ES: 'M'}, prepared.X_work)
        representative = next(step for step in steps if step.var_name == 'v')
        derived = composed[-1]
        assert derived.model is representative.model
        assert derived.feature_cols == representative.feature_cols
        assert derived.materialization == representative.materialization
        assert (derived.scale_factor, derived.fit_scale_factor) == (1.0, 1.0)
        # Le plan, état du fit, ne reçoit jamais d'étape dérivée
        assert derived not in imputer.imputation_plan_.steps

    def test_anchored_derived_step_takes_the_scale_of_its_group(
        self, mixed_freq_panel_multifrequency
    ):
        """An anchored derived step is rescaled like a plan step of the same source frequency."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        runner, prepared = _replay_runner(
            imputer, _with_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        )
        steps = imputer.imputation_plan_.by_stage()['M']
        composed = replay_steps(runner, steps, [ES], {**_MONTHLY, ES: 'M'}, prepared.X_work)
        plan_step = next(s for s in steps if s.var_name == 'v' and s.source_frequency == 'Q')
        assert composed[-1].scale_factor == plan_step.scale_factor == 3.0
        assert composed[-1].fit_scale_factor is composed[-1].scale_factor

    def test_pair_unanchored_at_fit_and_observed_now_is_anchored(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity of a fit-unanchored step that the transform observes leaves it for an anchored one."""
        blank = mixed_freq_panel_multifrequency.copy()
        blank.loc['DE', 'v'] = np.nan
        imputer = _fitted(_imputer(impute_unobserved_entities=True), blank)
        plan = [
            (s.var_name, s.source_frequency, s.entities, s.unanchored)
            for s in imputer.imputation_plan_
        ]
        assert ('v', None, (DE,), True) in plan

        composed = _monthly_replay(imputer, mixed_freq_panel_multifrequency)
        # Étape sans ancre vidée de DE, donc retirée ; DE servie par une étape ancrée
        assert ('v', None, (DE,), True) not in composed
        assert composed[-1] == ('v', 'Q', (DE,), False)

    def test_misaligned_new_pair_is_left_untouched(self, mixed_freq_panel_multifrequency):
        """A new pair off its calendar bounds gets neither an anchored nor an unanchored step."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency, source='DE')
        values = np.full(36, np.nan)
        values[1::3] = 5.0
        data.loc['ES', 'v'] = values
        composed = _monthly_replay(imputer, data)
        assert [step for step in composed if step[0] == 'v' and ES in step[2]] == []


# =============================================================================
# Rejeu du plan
# =============================================================================
class TestReplayPlan:
    """The whole plan replayed stage by stage, through ``StageRunner.replay_step``."""

    @staticmethod
    def _replay(imputer, data, *, log=None, skip_columns=frozenset()):
        """Replay the plan of ``imputer`` on ``data``, returning frames, provenances and runner."""
        runner, prepared = _replay_runner(imputer, data, log=log)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with runner.run.aggregation_constraint.deferred_warnings():
                frames, provenances = replay_plan(
                    runner, imputer.imputation_plan_, prepared.X_work,
                    entities=imputer.entities_, skip_columns=skip_columns,
                )
        return frames, provenances, runner

    @pytest.mark.parametrize(
        'overrides',
        [{}, {'covariate_strategy': 'model', 'impute_intermediate_frequencies': True}],
    )
    def test_frames_are_those_of_transform(self, panel_reference_full, overrides):
        """The target frame and provenance are, bit for bit, those ``transform`` returns."""
        imputer = _fitted(
            _imputer(keep_lower_frequencies=False, **overrides), panel_reference_full
        )
        frames, provenances, _ = self._replay(imputer, panel_reference_full)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = imputer.transform(panel_reference_full)
        assert list(frames) == [
            stage_frequency_label(stage) for stage in imputer.frequency_progression_
        ]
        pd.testing.assert_frame_equal(frames['M'], output, check_exact=True)
        pd.testing.assert_frame_equal(
            provenances['M'], imputer.imputation_provenance_, check_exact=True
        )

    def test_stages_follow_the_progression(self, reference_timeseries):
        """One frame and one provenance per stage, keyed by label, in progression order."""
        imputer = _fitted(_imputer(impute_intermediate_frequencies=True), reference_timeseries)
        frames, provenances, _ = self._replay(imputer, reference_timeseries)
        assert list(frames) == list(provenances) == ['Q', 'M']
        assert all(
            frames[label].index.equals(provenances[label].index) for label in frames
        )

    def test_every_step_goes_through_replay_step(self, mixed_freq_panel_multifrequency, monkeypatch):
        """Each plan step is executed once, in plan order, by the execution of the fit."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        executed = []
        original = StageRunner.replay_step

        def _spy(self, step, *, X_work, stage_freq):
            executed.append(step)
            return original(self, step, X_work=X_work, stage_freq=stage_freq)

        monkeypatch.setattr(StageRunner, 'replay_step', _spy)
        self._replay(imputer, mixed_freq_panel_multifrequency)
        assert executed == list(imputer.imputation_plan_.steps)

    def test_skipped_columns_are_not_replayed(self, mixed_freq_panel_multifrequency, monkeypatch):
        """The steps of a skipped column are dropped before any execution."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        executed = []
        original = StageRunner.replay_step

        def _spy(self, step, *, X_work, stage_freq):
            executed.append(step.var_name)
            return original(self, step, X_work=X_work, stage_freq=stage_freq)

        monkeypatch.setattr(StageRunner, 'replay_step', _spy)
        self._replay(imputer, mixed_freq_panel_multifrequency, skip_columns=frozenset({'v'}))
        assert executed == ['q1']

    def test_new_entity_left_untouched_is_noted(self, mixed_freq_panel_multifrequency):
        """Under ``impute_unobserved_entities=False`` a new entity keeps its input and is named."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency)
        frames, _, runner = self._replay(imputer, data)
        notes = runner.run.pending_warnings.messages()
        assert any(
            "1 entity/entities absent from the fit (('ES',),) are left untouched "
            "(impute_unobserved_entities=False)" in note
            for note in notes
        )
        pd.testing.assert_series_equal(
            frames['M'].loc['ES', 'v'], data.loc['ES', 'v'],
            check_names=False, check_freq=False,
        )

    def test_misaligned_new_pair_is_noted(self, mixed_freq_panel_multifrequency):
        """Under ``impute_unobserved_entities=True`` a new pair off its bounds is named."""
        imputer = _fitted(_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency)
        data = _with_entity(mixed_freq_panel_multifrequency, source='DE')
        values = np.full(36, np.nan)
        values[1::3] = 5.0
        data.loc['ES', 'v'] = values
        _, _, runner = self._replay(imputer, data)
        assert any(
            "1 (entity, column) pair(s) unknown to the fit are observed off the "
            "calendar bounds of their frequency and are left untouched: "
            "[('ES', 'v')]" in note
            for note in runner.run.pending_warnings.messages()
        )

    def test_each_stage_is_logged(self, reference_timeseries):
        """In verbose mode, one line per stage names the number of steps replayed."""
        imputer = _fitted(_imputer(impute_intermediate_frequencies=True), reference_timeseries)
        logged = []
        self._replay(imputer, reference_timeseries, log=logged.append)
        stage_lines = [line for line in logged if line.startswith('[transform] stage ')]
        n_steps = {
            label: len(steps) for label, steps in imputer.imputation_plan_.by_stage().items()
        }
        assert stage_lines == [
            f"[transform] stage {label}: replaying {n_steps.get(label, 0)} step(s)"
            for label in ('Q', 'M')
        ]

    def test_working_frame_is_never_modified(self, mixed_freq_panel_multifrequency):
        """The imputations travel through the registers: the working frame stays as given."""
        imputer = _fitted(_imputer(), mixed_freq_panel_multifrequency)
        runner, prepared = _replay_runner(imputer, mixed_freq_panel_multifrequency)
        before = prepared.X_work.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            replay_plan(
                runner, imputer.imputation_plan_, prepared.X_work, entities=imputer.entities_
            )
        pd.testing.assert_frame_equal(prepared.X_work, before, check_exact=True)
