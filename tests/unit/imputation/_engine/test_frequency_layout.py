"""Tests of ``FrequencyLayout`` and ``plan_frequencies`` in ``tsforecast.imputation._engine.frequency_layout``.

Scenarios: detection per (entity, column) pair on PANEL-X, PANEL-F and the realistic
heterogeneous-coverage panel (undetected and misaligned pairs, both shapes of the frequencies
by column), expansion and validation of the target frequency, classification, imputable groups
and pairs without anchor, progression per modality and merged over heterogeneous targets, stage
labels and per-row stage frequency, and the layout the imputer publishes its attributes from.
"""

# Modules de base
import dataclasses
import warnings

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.imputation import HighFrequencyImputer
from tsforecast.imputation._engine.frequency_layout import (
    FrequencyLayout,
    build_frequency_progression,
    classify_variables_at_frequency,
    expand_target_frequency,
    misaligned_anchor_pairs,
    plan_frequencies,
    stage_frequency_label,
    stage_frequency_of,
)
from tsforecast.panel.utils import get_unique_panel_entities, is_panel_data

# Clés d'entité des jeux de référence
FR, DE, IT = ('FR',), ('DE',), ('IT',)
# Clés d'entité du panel réaliste à couverture hétérogène
FRANCE, ALLEMAGNE, ITALIE = ('France',), ('Allemagne',), ('Italie',)


# Calcul du plan des fréquences d'un jeu, structure déduite des données
def _plan(
    data: pd.DataFrame,
    target_frequency='M',
    *,
    modality=False,
    on_frequency_mismatch='error',
    log=None,
) -> FrequencyLayout:
    """Compute the layout of a frame as the fit does.

    Args:
        data: Time series or panel.
        target_frequency: Target frequency, a string or a dict keyed by
            entity tuples.
        modality: Value of ``impute_intermediate_frequencies``.
        on_frequency_mismatch: Mismatch policy of the target validation.
        log: Logging callback.

    Returns:
        The layout.
    """
    is_panel = is_panel_data(data=data)
    return plan_frequencies(
        data,
        target_frequency,
        is_panel=is_panel,
        entities=get_unique_panel_entities(data) if is_panel else None,
        impute_intermediate_frequencies=modality,
        on_frequency_mismatch=on_frequency_mismatch,
        log=log,
    )


# Effacement d'une colonne pour une entité
def _erase(data: pd.DataFrame, entity: str, column: str) -> pd.DataFrame:
    """Return a copy of a panel where one entity never observes one column.

    Args:
        data: Panel indexed by (entity, date).
        entity: Entity label.
        column: Column to erase.

    Returns:
        The modified copy.
    """
    erased = data.copy()
    erased.loc[entity, column] = np.nan
    return erased


class TestPanelX:
    """Layout of the unified reference panel PANEL-X."""

    def test_frequencies_are_detected_per_pair(self, panel_reference_full):
        """The same column carries its own frequency for each entity."""
        layout = _plan(panel_reference_full)

        assert layout.is_panel
        assert layout.detected_frequencies[('FR', 'v')] == 'Y'
        assert layout.detected_frequencies[('DE', 'v')] == 'Q'
        assert layout.detected_frequencies[('IT', 'v')] == 'M'
        for entity in ('FR', 'DE', 'IT'):
            assert layout.detected_frequencies[(entity, 'm1')] == 'M'
            assert layout.detected_frequencies[(entity, 'q1')] == 'Q'
            assert layout.detected_frequencies[(entity, 'a1')] == 'Y'

    def test_never_observed_pair_is_undetected(self, panel_reference_full):
        """``climat_affaires``, never observed by IT, is set aside, never detected."""
        layout = _plan(panel_reference_full)

        assert layout.undetected == (('IT', 'climat_affaires'),)
        assert ('IT', 'climat_affaires') not in layout.detected_frequencies
        assert layout.misaligned == ()

    def test_string_target_is_bound_to_every_entity(self, panel_reference_full):
        """A string target becomes a binding naming every entity."""
        layout = _plan(panel_reference_full)

        assert layout.target_frequency == {DE: 'M', FR: 'M', IT: 'M'}

    def test_categories_are_per_pair(self, panel_reference_full):
        """``v`` is imputable for FR and DE, already at target for IT."""
        layout = _plan(panel_reference_full)

        assert ('FR', 'v') in layout.categories['impute']
        assert ('DE', 'v') in layout.categories['impute']
        assert ('IT', 'v') in layout.categories['target_freq']
        assert layout.categories['aggregate'] == []
        # Couple indétectable : dans aucune catégorie
        assert all(
            ('IT', 'climat_affaires') not in keys for keys in layout.categories.values()
        )

    @pytest.mark.parametrize('modality', ['covariates_only', True])
    def test_complete_progression_under_both_cascading_modalities(
        self, panel_reference_full, modality
    ):
        """``'covariates_only'`` and ``True`` share the very same stage plan."""
        layout = _plan(panel_reference_full, modality=modality)

        assert layout.progression == [
            {DE: 'Q', FR: 'Q', IT: 'Q'},
            {DE: 'M', FR: 'M', IT: 'M'},
        ]

    def test_single_target_stage_without_cascade(self, panel_reference_full):
        """Under ``False`` the progression is the target binding alone."""
        layout = _plan(panel_reference_full, modality=False)

        assert layout.progression == [layout.target_frequency]

    def test_unanchored_pair_travels_every_stage(self, panel_reference_full):
        """IT's ``climat_affaires`` joins a ``(column, None)`` group at every stage."""
        layout = _plan(panel_reference_full, modality=True)

        for stage in layout.progression:
            groups = layout.imputable_groups(stage, impute_unobserved_entities=True)
            assert groups[('climat_affaires', None)] == (IT,)
            # Sans la capacité, le groupe sans ancre n'existe pas, les autres
            # sont inchangés
            closed = layout.imputable_groups(stage, impute_unobserved_entities=False)
            assert ('climat_affaires', None) not in closed
            assert closed == {
                key: entities for key, entities in groups.items() if key[1] is not None
            }

    def test_groups_per_source_frequency_at_each_stage(self, panel_reference_full):
        """Stage Q imputes the annual pairs, stage M every coarser pair."""
        layout = _plan(panel_reference_full, modality=True)
        stage_q, stage_m = layout.progression

        assert layout.imputable_groups(stage_q, impute_unobserved_entities=False) == {
            ('a1', 'Y'): (DE, FR, IT),
            ('a2', 'Y'): (DE, FR, IT),
            ('v', 'Y'): (FR,),
        }
        assert layout.imputable_groups(stage_m, impute_unobserved_entities=False) == {
            ('q1', 'Q'): (DE, FR, IT),
            ('a1', 'Y'): (DE, FR, IT),
            ('a2', 'Y'): (DE, FR, IT),
            ('v', 'Q'): (DE,),
            ('v', 'Y'): (FR,),
        }


class TestPanelF:
    """Layout of PANEL-F, whose column ``v`` is annual, quarterly and monthly by entity."""

    def test_frequencies_by_column_keep_both_shapes(self, mixed_freq_panel_multifrequency):
        """A heterogeneous column stays per entity, the others fold to a scalar."""
        by_column = _plan(mixed_freq_panel_multifrequency).frequencies_by_column()

        # Colonne hétérogène : forme par entité
        assert by_column['v'] == {FR: 'Y', DE: 'Q', IT: 'M'}
        # Colonnes homogènes : repli sur la forme scalaire
        assert by_column['q1'] == 'Q'
        assert by_column['m1'] == 'M'

    def test_transform_frequencies_never_override_the_fit(self, mixed_freq_panel_multifrequency):
        """A pair of the fit keeps its frequency; a new pair adds its own."""
        layout = _plan(mixed_freq_panel_multifrequency)

        by_column = layout.frequencies_by_column({('FR', 'v'): 'M', ('ES', 'v'): 'Q'})

        assert by_column['v'] == {FR: 'Y', DE: 'Q', IT: 'M', ('ES',): 'Q'}

    def test_imputable_groups_per_source_frequency(self, mixed_freq_panel_multifrequency):
        """At stage M: ``(v, Y) -> {FR}``, ``(v, Q) -> {DE}``, IT in no group of ``v``."""
        layout = _plan(mixed_freq_panel_multifrequency)
        (stage,) = layout.progression

        assert layout.imputable_groups(stage, impute_unobserved_entities=True) == {
            ('q1', 'Q'): (DE, FR, IT),
            ('v', 'Y'): (FR,),
            ('v', 'Q'): (DE,),
        }

    def test_entity_erased_from_a_column_gets_an_unanchored_group(
        self, mixed_freq_panel_multifrequency
    ):
        """IT never observing ``v``: undetected, unclassified, imputable only on demand."""
        layout = _plan(_erase(mixed_freq_panel_multifrequency, 'IT', 'v'))
        (stage,) = layout.progression

        assert layout.undetected == (('IT', 'v'),)
        assert sorted(key for key in layout.categories['impute'] if key[1] == 'v') == [
            ('DE', 'v'), ('FR', 'v'),
        ]
        groups = layout.imputable_groups(stage, impute_unobserved_entities=True)
        assert groups[('v', None)] == (IT,)
        assert ('v', None) not in layout.imputable_groups(
            stage, impute_unobserved_entities=False
        )

    def test_unanchored_pairs_need_another_observer(self, mixed_freq_panel_multifrequency):
        """A column no entity observes has no model to share: no unanchored pair."""
        data = mixed_freq_panel_multifrequency.assign(w=np.nan)
        layout = _plan(data)
        (stage,) = layout.progression

        assert {key[1] for key in layout.undetected} == {'w'}
        assert layout.unanchored_pairs_at(stage, impute_unobserved_entities=True) == []

    def test_unanchored_pairs_require_a_stage_binding(self, mixed_freq_panel_multifrequency):
        """A stage that does not bind the entity never makes its pair imputable."""
        layout = _plan(_erase(mixed_freq_panel_multifrequency, 'IT', 'v'))

        assert layout.unanchored_pairs_at(
            {FR: 'M', DE: 'M'}, impute_unobserved_entities=True
        ) == []
        assert layout.unanchored_pairs_at(
            {IT: 'M'}, impute_unobserved_entities=True
        ) == [(IT, 'v')]

    def test_training_frequencies_name_the_undetected_entity(
        self, mixed_freq_panel_multifrequency
    ):
        """The training set never borrows a frequency for an undetectable pair."""
        data = _erase(mixed_freq_panel_multifrequency, 'IT', 'v')
        # DE annuelle comme FR : la colonne se replie sur la forme scalaire
        data.loc['DE', 'v'] = data.loc['FR', 'v'].to_numpy()
        layout = _plan(data)
        by_column = layout.frequencies_by_column()

        assert by_column['v'] == 'Y'
        explicit = layout.training_frequencies('v', by_column)
        assert explicit['v'] == {FR: 'Y', DE: 'Y', IT: None}
        # Covariables intactes, et rien à expliciter pour une colonne complète
        assert explicit['q1'] == by_column['q1']
        assert layout.training_frequencies('q1', by_column) is by_column


class TestMergedProgression:
    """Progressions per target-frequency group, merged into global stages."""

    def test_heterogeneous_targets_on_panel_f(self, mixed_freq_panel_multifrequency):
        """Targets ``{FR: M, DE: Q, IT: M}``: DE travels stage Q only."""
        layout = _plan(
            mixed_freq_panel_multifrequency,
            {FR: 'M', DE: 'Q', IT: 'M'},
            modality=True,
        )

        assert layout.progression == [{FR: 'Q', DE: 'Q', IT: 'Q'}, {FR: 'M', IT: 'M'}]
        # Étape M : DE, absente de la liaison, n'est classée nulle part
        stage_m = layout.progression[1]
        classified = layout.classify(stage_m)
        assert all(key[0] != 'DE' for keys in classified.values() for key in keys)
        groups = layout.imputable_groups(stage_m, impute_unobserved_entities=True)
        assert all(DE not in entities for entities in groups.values())

    def test_two_groups_of_different_depth(self):
        """``F(FR) = {Y, Q, M}`` and ``F(DE) = {Y, Q}`` merge into ``[{FR, DE}: Q, {FR}: M]``."""
        detected = {
            ('FR', 'm1'): 'M', ('FR', 'q1'): 'Q', ('FR', 'v'): 'Y',
            ('DE', 'q1'): 'Q', ('DE', 'v'): 'Y',
        }
        target = {FR: 'M', DE: 'Q'}
        imputable = classify_variables_at_frequency(detected, target, is_panel=True)['impute']

        progression = build_frequency_progression(
            detected, imputable, target, impute_intermediate_frequencies=True
        )

        assert progression == [{FR: 'Q', DE: 'Q'}, {FR: 'M'}]

    def test_lowest_frequency_is_never_a_stage(self, reference_timeseries):
        """With annual sources only, the cascade has the target stage alone."""
        layout = _plan(reference_timeseries.drop(columns='q1'), modality=True)

        assert layout.progression == ['M']

    @pytest.mark.parametrize(
        ('modality', 'expected'),
        [(False, ['M']), ('covariates_only', ['Q', 'M']), (True, ['Q', 'M'])],
    )
    def test_time_series_progression(self, reference_timeseries, modality, expected):
        """``TS``: Y is the lowest frequency, never a stage."""
        layout = _plan(reference_timeseries, modality=modality)

        assert not layout.is_panel
        assert layout.target_frequency == 'M'
        assert layout.progression == expected

    def test_time_series_frequencies_by_column(self, reference_timeseries):
        """On a time series the keys are already columns: a copy, transform pairs added."""
        layout = _plan(reference_timeseries)

        by_column = layout.frequencies_by_column({'new': 'Q', 'a1': 'M'})

        assert by_column == {'new': 'Q', 'm1': 'M', 'q1': 'Q', 'a1': 'Y', 'a2': 'Y'}
        assert layout.frequencies_by_column() is not layout.detected_frequencies


class TestHeterogeneousCoveragePanel:
    """Layout of the realistic panel with per-entity coverage and publication frequency."""

    def test_publication_frequency_differs_by_entity(self, heterogeneous_coverage_panel):
        """``depenses_publiques_pib`` is annual for France and Italie, quarterly for Allemagne."""
        layout = _plan(heterogeneous_coverage_panel)

        assert layout.column_frequencies_by_entity('depenses_publiques_pib') == {
            ALLEMAGNE: 'Q', FRANCE: 'Y', ITALIE: 'Y',
        }
        assert layout.frequencies_by_column()['depenses_publiques_pib'] == {
            ALLEMAGNE: 'Q', FRANCE: 'Y', ITALIE: 'Y',
        }

    def test_structurally_absent_column_is_undetected(self, heterogeneous_coverage_panel):
        """``climat_affaires``, never observed by Italie, is its single undetected pair."""
        layout = _plan(heterogeneous_coverage_panel)

        assert layout.undetected == (('Italie', 'climat_affaires'),)
        # Les entités qui l'observent s'accordent : forme scalaire
        assert layout.frequencies_by_column()['climat_affaires'] == 'M'
        # Le jeu d'entraînement ne prête pas cette fréquence à Italie
        explicit = layout.training_frequencies(
            'climat_affaires', layout.frequencies_by_column()
        )
        assert explicit['climat_affaires'] == {ALLEMAGNE: 'M', FRANCE: 'M', ITALIE: None}

    def test_cascade_travels_the_quarter(self, heterogeneous_coverage_panel):
        """Quarterly and annual sources: a Q stage, then the target M."""
        layout = _plan(heterogeneous_coverage_panel, modality=True)

        assert [stage_frequency_label(stage) for stage in layout.progression] == ['Q', 'M']
        stage_q = layout.progression[0]
        groups = layout.imputable_groups(stage_q, impute_unobserved_entities=False)
        # À l'étape Q : les seules sources annuelles, Allemagne publiant
        # trimestriellement ses dépenses publiques
        assert groups[('depenses_publiques_pib', 'Y')] == (FRANCE, ITALIE)
        assert ('depenses_publiques_pib', 'Q') not in groups
        assert ('pib_trimestriel', 'Q') not in groups


class TestTargetFrequency:
    """Expansion and validation of the target frequency."""

    def test_incomplete_dict_names_the_missing_entity(self, mixed_freq_panel_multifrequency):
        """A dict target missing an entity raises, naming it."""
        with pytest.raises(ValueError, match=r"incomplete.*\('IT',\)"):
            _plan(mixed_freq_panel_multifrequency, {FR: 'M', DE: 'Q'})

    def test_dict_target_on_a_time_series_is_refused(self, reference_timeseries):
        """A dict target on a time series raises a ValueError naming the remedy, before any detection."""
        with pytest.raises(ValueError, match='cannot be a dict for simple time series'):
            expand_target_frequency({FR: 'M'}, is_panel=False, entities=None)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError, match='Use a string frequency instead'):
                _plan(reference_timeseries, {FR: 'M'})

    def test_imputer_refuses_a_dict_target_on_a_time_series(self, reference_timeseries):
        """At fit, a dict target on a time series is a ValueError, never a TypeError."""
        imputer = HighFrequencyImputer(target_frequency={'FR': 'M'}, estimator=LinearRegression())

        with pytest.raises(ValueError, match='cannot be a dict for simple time series'):
            imputer.fit(reference_timeseries)

    def test_dict_target_is_copied(self):
        """The binding is a copy: the caller's dict is never shared."""
        target = {FR: 'M'}

        expanded = expand_target_frequency(target, is_panel=True, entities=[FR])

        assert expanded == target
        assert expanded is not target

    def test_entity_without_any_detected_pair_leaves_the_target(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity entirely NaN is withdrawn from the target, not reported missing."""
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['IT', :] = np.nan

        layout = _plan(data, {FR: 'M', DE: 'M', IT: 'M'})

        assert layout.target_frequency == {FR: 'M', DE: 'M'}
        assert layout.progression == [{FR: 'M', DE: 'M'}]

    def test_entity_target_frequency(self, mixed_freq_panel_multifrequency, reference_timeseries):
        """The target of one entity, normalized; None for an entity the binding omits."""
        panel = _plan(mixed_freq_panel_multifrequency, {FR: 'ME', DE: 'QE', IT: 'M'})
        series = _plan(reference_timeseries)

        assert panel.entity_target_frequency(FR) == 'M'
        assert panel.entity_target_frequency(DE) == 'Q'
        assert panel.entity_target_frequency(('ES',)) is None
        assert series.entity_target_frequency(None) == 'M'


class TestDetection:
    """Robust detection: undetectable and misaligned pairs."""

    def test_no_detectable_frequency_raises(self):
        """A frame with no detectable pair raises."""
        dates = pd.date_range('2021-01-31', periods=3, freq='ME')
        data = pd.DataFrame({'a': [1.0, np.nan, np.nan]}, index=dates)

        with pytest.raises(ValueError, match='Could not detect frequency for any column'):
            _plan(data)

    def test_undetected_pairs_are_logged(self, panel_reference_full):
        """The log callback names the undetected pairs."""
        messages = []

        _plan(panel_reference_full, log=messages.append)

        assert len(messages) == 1
        assert "('IT', 'climat_affaires')" in messages[0]

    def test_nothing_logged_without_undetected_pair(self, mixed_freq_panel_multifrequency):
        """No undetected pair, no message."""
        messages = []

        _plan(mixed_freq_panel_multifrequency, log=messages.append)

        assert messages == []

    def test_fiscal_year_is_left_out_under_one_warning(self, reference_timeseries):
        """A yearly column closing in June is misaligned: warned, neither detected nor classified."""
        data = reference_timeseries.copy()
        dates = data.index
        data['fy'] = np.nan
        data.loc[dates[5::12], 'fy'] = [1.0, 2.0, 3.0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            layout = _plan(data)

        messages = [str(w.message) for w in caught if 'off the calendar bounds' in str(w.message)]
        assert len(messages) == 1
        assert "['fy']" in messages[0]
        assert layout.misaligned == ('fy',)
        assert 'fy' not in layout.detected_frequencies
        assert all('fy' not in keys for keys in layout.categories.values())

    def test_misaligned_pair_is_explicit_for_the_training_set(
        self, mixed_freq_panel_multifrequency
    ):
        """On a panel, a misaligned pair is mapped to None like an undetectable one."""
        data = mixed_freq_panel_multifrequency.copy()
        fr_dates = data.loc['FR'].index
        data.loc['FR', 'v'] = np.nan
        data.loc[[('FR', date) for date in fr_dates[5::12]], 'v'] = [1.0, 2.0, 3.0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            layout = _plan(data)

        assert layout.misaligned == (('FR', 'v'),)
        explicit = layout.training_frequencies('v', layout.frequencies_by_column())
        assert explicit['v'] == {DE: 'Q', IT: 'M', FR: None}

    @pytest.mark.parametrize(('month', 'misaligned'), [(1, ()), (12, ('a',))])
    def test_period_start_index_expects_the_first_day(self, month, misaligned):
        """On an ``MS`` index a yearly value belongs on 1 January, not on 1 December."""
        dates = pd.date_range('2021-01-01', periods=36, freq='MS')
        data = pd.DataFrame({'m': np.arange(36.0), 'a': np.nan}, index=dates)
        data.loc[dates[dates.month == month], 'a'] = [1.0, 2.0, 3.0]

        assert misaligned_anchor_pairs(
            data, {'m': 'M', 'a': 'Y'}, is_panel=False
        ) == misaligned

    def test_index_without_position_is_not_checked(self):
        """A daily index implies no calendar bound."""
        dates = pd.date_range('2021-01-01', periods=60, freq='D')
        data = pd.DataFrame({'d': np.arange(60.0), 'w': np.nan}, index=dates)
        data.loc[dates[3::7], 'w'] = 1.0

        assert misaligned_anchor_pairs(data, {'d': 'D', 'w': 'W'}, is_panel=False) == ()

    @pytest.mark.parametrize(
        'dates',
        [
            ['2021-06-30'],
            ['2021-01-31', '2021-02-03', '2021-05-17', '2021-11-30'],
        ],
        ids=['single_row', 'irregular'],
    )
    def test_index_without_determinable_frequency_is_not_checked(self, dates):
        """An index too short or too irregular for a frequency implies no calendar bound."""
        data = pd.DataFrame({'a': np.arange(float(len(dates)))}, index=pd.DatetimeIndex(dates))

        assert misaligned_anchor_pairs(data, {'a': 'Y'}, is_panel=False) == ()

    def test_non_calendar_frequency_is_not_checked(self, reference_timeseries):
        """Only monthly, quarterly and yearly pairs have calendar bounds to check."""
        assert misaligned_anchor_pairs(
            reference_timeseries, {'m1': 'D', 'q1': 'W'}, is_panel=False
        ) == ()

    def test_pair_without_frequency_is_not_checked(self, reference_timeseries):
        """An undetectable pair has no period to be misaligned with."""
        data = reference_timeseries.assign(once=np.nan)
        data.loc[data.index[5], 'once'] = 1.0

        assert misaligned_anchor_pairs(
            data, {'m1': 'M', 'once': None}, is_panel=False
        ) == ()


class TestClassification:
    """Classification of the pairs relative to a prediction frequency."""

    def test_time_series_three_categories(self):
        """Finer, equal and coarser columns are aggregated, kept and imputed."""
        result = classify_variables_at_frequency(
            {'d': 'D', 'm': 'ME', 'q': 'Q'}, 'M', is_panel=False
        )

        assert result == {'aggregate': ['d'], 'impute': ['q'], 'target_freq': ['m']}

    def test_entity_absent_from_the_binding_is_skipped(self):
        """A panel entity the stage does not bind is classified nowhere."""
        result = classify_variables_at_frequency(
            {('FR', 'v'): 'Y', ('DE', 'v'): 'Y'}, {FR: 'M'}, is_panel=True
        )

        assert result == {'aggregate': [], 'impute': [('FR', 'v')], 'target_freq': []}


class TestStageHelpers:
    """Stage labels and per-row stage frequency."""

    @pytest.mark.parametrize(
        ('stage', 'label'),
        [
            ('M', 'M'),
            ({FR: 'QE', DE: 'Q'}, 'Q'),
            ({FR: 'M', DE: 'Q', IT: 'M'}, "('DE',)=Q+('FR',)=M+('IT',)=M"),
        ],
    )
    def test_stage_frequency_label(self, stage, label):
        """A shared frequency is the label; divergent ones give a sorted composite."""
        assert stage_frequency_label(stage) == label

    def test_stage_frequency_of_time_series(self):
        """A scalar stage gives the same base code on every row, ``str`` dtype."""
        index = pd.date_range('2021-01-31', periods=3, freq='ME')

        result = stage_frequency_of('ME', index)

        assert result.tolist() == ['M', 'M', 'M']
        assert result.dtype == 'str'
        pd.testing.assert_index_equal(result.index, index)

    def test_stage_frequency_of_panel_follows_the_entity(self):
        """A binding gives each row its entity's base code, rows in input order."""
        dates = pd.date_range('2021-03-31', periods=2, freq='QE')
        index = pd.MultiIndex.from_tuples(
            [('DE', dates[0]), ('FR', dates[0]), ('DE', dates[1]), ('FR', dates[1])]
        )

        result = stage_frequency_of({FR: 'QE', DE: 'monthly'}, index)

        assert result.tolist() == ['M', 'Q', 'M', 'Q']
        assert result.dtype == 'str'


class TestLayoutObject:
    """The layout itself."""

    def test_layout_is_frozen(self, mixed_freq_panel_multifrequency):
        """No field of a computed layout can be reassigned."""
        layout = _plan(mixed_freq_panel_multifrequency)

        with pytest.raises(dataclasses.FrozenInstanceError):
            layout.progression = []

    def test_layout_does_not_depend_on_the_column_order(self, panel_reference_full):
        """Reversing the columns changes no frequency, group nor stage."""
        layout = _plan(panel_reference_full, modality=True)
        reversed_layout = _plan(panel_reference_full[panel_reference_full.columns[::-1]], modality=True)

        assert reversed_layout.detected_frequencies == layout.detected_frequencies
        assert reversed_layout.progression == layout.progression
        for stage in layout.progression:
            assert reversed_layout.imputable_groups(
                stage, impute_unobserved_entities=True
            ).items() == layout.imputable_groups(stage, impute_unobserved_entities=True).items()


class TestImputerPublishesTheLayout:
    """``HighFrequencyImputer`` publishes its fitted attributes from one layout."""

    # Interne : attribut privé ; le contenu public est figé par le golden master
    @pytest.mark.internal
    @pytest.mark.parametrize('modality', [False, True])
    def test_fitted_attributes_are_the_layout_fields(self, panel_reference_full, modality):
        """The public attributes are the very objects of the layout, computed once."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            imputer = HighFrequencyImputer(
                target_frequency='M',
                estimator=LinearRegression(),
                impute_intermediate_frequencies=modality,
            ).fit(panel_reference_full)
        layout = imputer._frequency_layout_

        assert imputer.detected_frequencies_ is layout.detected_frequencies
        assert imputer.effective_target_frequency_ is layout.target_frequency
        assert imputer.variable_categories_ is layout.categories
        assert imputer.frequency_progression_ is layout.progression
        assert layout == _plan(panel_reference_full, modality=modality)
