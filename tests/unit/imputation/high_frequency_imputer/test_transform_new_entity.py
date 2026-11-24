"""``HighFrequencyImputer.transform`` on an entity absent from the fit.

Scenarios: new entities and pairs unknown to the fit met at transform.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.imputation._components.provenance import ProvenanceType

# Outils de test partagés
from tests.unit.imputation.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
    transform_quietly,
    _warning_messages,
    _with_new_entity,
)


class TestNewEntityAtTransform:
    """An entity absent from the fit, met at transform."""

    # Fixture privée d'une entité ajoutée après le fit
    @staticmethod
    def _with_new_entity(data: pd.DataFrame) -> pd.DataFrame:
        """PANEL-F augmenté d'une entité ES, dont 'v' n'est jamais observée."""
        added = data.xs('IT', level=0, drop_level=False).copy()
        added.index = pd.MultiIndex.from_arrays(
            [
                ['ES'] * len(added),
                added.index.get_level_values(-1),
            ],
            names=data.index.names,
        )
        added['v'] = np.nan
        return pd.concat([data, added]).sort_index()

    @pytest.mark.slow
    def test_new_entity_is_left_untouched_by_default(
        self, mixed_freq_panel_multifrequency
    ):
        """Sous False, rien ne lui est écrit et un avertissement la nomme."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        extended = self._with_new_entity(mixed_freq_panel_multifrequency)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(extended)

        messages = [str(w.message) for w in caught if 'absent from the fit' in str(w.message)]
        assert len(messages) == 1
        assert 'ES' in messages[0]

        # Les 36 cellules de 'v' restent vides
        target = result.xs('M', level='frequency')
        assert target.xs('ES', level=0)['v'].isna().all()
        # Les autres entités sont intactes
        reference = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        for entity in ('FR', 'DE', 'IT'):
            pd.testing.assert_series_equal(
                result.xs('M', level='frequency').xs(entity, level=0)['v'],
                reference.xs('M', level='frequency').xs(entity, level=0)['v'],
            )

    def test_new_entity_is_imputed_on_demand(self, mixed_freq_panel_multifrequency):
        """Sous True, elle est imputée sans ancre, par le modèle du plan."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True),
            mixed_freq_panel_multifrequency,
        )
        extended = self._with_new_entity(mixed_freq_panel_multifrequency)
        result = transform_quietly(imputer, extended)

        target = result.xs('M', level='frequency').xs('ES', level=0)
        # Les 36 cellules sont renseignées
        assert target['v'].notna().sum() == 36
        # Elles portent la provenance des cellules sans ancre
        provenance = (
            imputer.imputation_provenance_
            .xs('M', level='frequency').xs('ES', level=0)['v']
        )
        assert set(provenance.dropna().unique()) == {ProvenanceType.MODEL_UNANCHORED}

    @pytest.mark.slow
    def test_new_entity_is_imputed_as_if_present_at_fit(self, mixed_freq_panel_multifrequency):
        """A new entity is served like an unobserved pair at fit: same values, same provenance.

        ``ES`` copies ``IT`` without ``v``. Present at fit time, it is an
        unobserved pair; born at transform time, it is served by
        derived steps. Since it contributes nothing to the pooled training
        set either way (R1), both routes give it the very same 36 cells.
        """
        extended = _with_new_entity(mixed_freq_panel_multifrequency)
        params = dict(impute_unobserved_entities=True, keep_lower_frequencies=False)

        # Voie de la paire non observée : ES présente au fit
        at_fit = make_imputer(**params)
        expected = fit_transform_quietly(at_fit, extended).loc['ES', 'v']
        # Voie des étapes dérivées : ES née au transform
        at_transform = fit_quietly(make_imputer(**params), mixed_freq_panel_multifrequency)
        actual = transform_quietly(at_transform, extended).loc['ES', 'v']

        pd.testing.assert_series_equal(actual, expected)
        assert set(at_transform.imputation_provenance_.loc['ES', 'v']) == {
            ProvenanceType.MODEL_UNANCHORED
        }

    def test_derived_steps_never_enter_the_plan(self, mixed_freq_panel_multifrequency):
        """The plan and ``unanchored_pairs_`` stay those of the fit after serving ES."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency
        )
        plan_before = imputer.imputation_plan_.to_diagnostic_frame()

        result = transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))

        # ES a bien été servie...
        assert result.xs('M', level='frequency').loc['ES', 'v'].notna().all()
        # ... sans qu'aucune étape dérivée n'entre au plan
        pd.testing.assert_frame_equal(imputer.imputation_plan_.to_diagnostic_frame(), plan_before)
        assert imputer.unanchored_pairs_ == ()

    def test_stage_with_diverging_frequencies_cannot_bind_a_new_entity(
        self, mixed_freq_panel_multifrequency
    ):
        """A stage whose entities disagree on its frequency leaves the new entity out.

        Targets ``{FR: M, DE: Q, IT: M}`` without axis 2: the single stage
        binds diverging frequencies, nothing departs them for ``ES``. One
        aggregated warning names the entity and the stage; its cells stay NaN
        and unmarked, even under ``impute_unobserved_entities=True``.
        """
        imputer = fit_quietly(
            make_imputer(
                target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
                impute_unobserved_entities=True, keep_lower_frequencies=False,
            ),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency)
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(extended))
        )

        skipped = [m for m in messages if 'disagree on the stage frequency' in m]
        assert len(skipped) == 1 and "('ES',)" in skipped[0]
        assert captured['result'].loc['ES', 'v'].isna().all()
        assert imputer.imputation_provenance_.loc['ES', 'v'].isna().all()

    def test_observations_of_a_new_entity_are_never_overwritten(
        self, mixed_freq_panel_multifrequency
    ):
        """The periods a new entity observes keep their observed totals.

        ``ES`` copies ``DE``, ``v`` and ``q1`` included: it observes both
        quarterly. It is served by anchored steps: the months
        of each observed quarter sum to the observation — under ``'sum'`` an
        anchor is re-expressed, never lost — and its cells carry the
        ``MODEL_*`` of the column's step, never ``MODEL_UNANCHORED``.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        result = transform_quietly(imputer, extended).loc['ES']
        source = extended.loc['ES']

        # Trimestres observés dont la somme mensuelle s'écarte de l'observation
        gaps = {}
        for column in ('v', 'q1'):
            sums = result[column].groupby(result.index.to_period('Q')).sum()
            observed = source[column].dropna()
            observed.index = observed.index.to_period('Q')
            gap = (sums.reindex(observed.index) - observed).abs()
            gaps[column] = int((gap > 1e-9).sum())
        assert gaps == {'v': 0, 'q1': 0}
        marks = imputer.imputation_provenance_.loc['ES', ['v', 'q1']]
        assert ProvenanceType.MODEL_UNANCHORED not in set(marks.stack())

    def test_new_entity_copy_is_imputed_like_its_model(self, mixed_freq_panel_multifrequency):
        """Same covariates, same frozen model, same totals: ``ES`` gets ``DE``'s cells.

        ``ES`` copies ``DE`` at transform time. Its anchored steps share the
        model of ``DE``'s and rescale on the same quarterly totals: the two
        entities receive the very same values and provenance.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())

        result = transform_quietly(imputer, extended)

        pd.testing.assert_frame_equal(result.loc['ES'], result.loc['DE'])
        marks = imputer.imputation_provenance_
        pd.testing.assert_frame_equal(marks.loc['ES'], marks.loc['DE'])

    def test_column_empty_at_fit_and_observed_at_transform_is_anchored(
        self, panel_f_without_it_v, mixed_freq_panel_multifrequency
    ):
        """A known entity's column the fit never observed is anchored once observed.

        ``IT`` leaves ``v`` empty at fit time: under ``True`` its pair is
        unanchored in the plan. At transform it publishes ``v`` quarterly,
        like ``DE``: the unanchored step would overwrite those observations,
        an anchored one rescales on them.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            panel_f_without_it_v,
        )
        assert imputer.unanchored_pairs_ == (('IT', 'v'),)
        published = panel_f_without_it_v.copy()
        published.loc[('IT', slice(None)), 'v'] = (
            mixed_freq_panel_multifrequency.loc['DE', 'v'].to_numpy()
        )

        result = transform_quietly(imputer, published).loc['IT', 'v']

        quarters = result.groupby(result.index.to_period('Q')).sum()
        observed = published.loc['IT', 'v'].dropna()
        np.testing.assert_allclose(quarters.to_numpy(), observed.to_numpy())
        marks = set(imputer.imputation_provenance_.loc['IT', 'v'])
        assert ProvenanceType.MODEL_UNANCHORED not in marks

    def test_new_pair_already_at_the_stage_frequency_is_left_as_given(
        self, mixed_freq_panel_multifrequency
    ):
        """A pair a new entity observes at the stage frequency has nothing to impute.

        ``ES`` copies ``IT``, which observes ``v`` monthly: at the monthly
        stage the pair is at target, as it would have been at fit time, and
        keeps its observations. Its quarterly ``q1`` is imputed and anchored.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='IT', clear=())

        result = transform_quietly(imputer, extended).loc['ES']

        pd.testing.assert_series_equal(result['v'], extended.loc['ES', 'v'], check_freq=False)
        assert result['q1'].notna().all()
        quarters = result['q1'].groupby(result.index.to_period('Q')).sum()
        np.testing.assert_allclose(
            quarters.to_numpy(), extended.loc['ES', 'q1'].dropna().to_numpy()
        )

    def test_new_pair_observed_off_its_calendar_is_left_as_given(
        self, mixed_freq_panel_multifrequency
    ):
        """A new pair dated off the bounds of its period is left out, as at fit.

        ``ES`` publishes ``v`` one month before each quarter end: a quarterly
        reading would attach each value to the wrong quarter. Neither
        anchored nor unanchored, the pair stays as given, and the note of the
        pairs outside the plan names it once.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        new_rows = extended.index.get_level_values(0) == 'ES'
        extended.loc[new_rows, 'v'] = extended.loc[new_rows, 'v'].shift(-1).to_numpy()
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(extended))
        )

        pd.testing.assert_series_equal(
            captured['result'].loc['ES', 'v'], extended.loc['ES', 'v'], check_freq=False
        )
        notes = [m for m in messages if 'off the calendar bounds' in m]
        assert len(notes) == 1 and "('ES', 'v')" in notes[0]

    def test_pairs_unknown_to_the_fit_stay_as_given_by_default(
        self, mixed_freq_panel_multifrequency
    ):
        """Under ``False`` a new entity is left as given, observed columns included."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False), mixed_freq_panel_multifrequency
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())

        result = transform_quietly(imputer, extended)

        pd.testing.assert_frame_equal(result.loc['ES'], extended.loc['ES'], check_freq=False)
        assert imputer.imputation_provenance_.loc['ES', 'v'].dropna().eq(
            ProvenanceType.ORIGINAL
        ).all()
