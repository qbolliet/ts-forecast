"""Tests of the four-rank materialization precedence.

Spec ``high_frequency_imputer2_architecture.md`` §4.4: when a covariate must
be materialized on the grid of stage ``f``, the precedence applies in order
and stops at the FIRST applicable rank — rank 1 identity / exact aggregation,
rank 2 the covariate's own production at ``f`` read from the mirror, rank 3
its production at an earlier stage ``f'`` carried onto ``f``, rank 4
``covariate_fallback``. Ranks 2 and 3 read the stores, hence exist under
``covariate_strategy='model'`` only (§8.1).

Each rank is first pinned on its own, then opposed to a lower rank that is
ALSO applicable, so that only the precedence decides — including rank 1
against the mirror, entity by entity (§4.6: "une imputation ne recouvre
jamais une mesure").

Covered symbols: ``CovariateMaterializer.classify``, ``materialize`` (model
ways), ``record_production`` (mirror priming).
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Jeux partagés
from tests.unit.frequency.covariate_materializer.helpers import (
    PANEL_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    SpyApplier,
    TS_FREQUENCIES,
    annual_grid,
    entity_grid,
    monthly_grid,
    panel_grid,
    prime,
    quarterly_grid,
)


class TestPrecedenceRanks:
    """§4.4: the four ranks, each on its own."""

    def test_rank_two_reads_stage_mirror(self, reference_timeseries):
        """Rank 2: the values come from the stage mirror, not from the source."""
        grid = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        mirror = pd.Series(7.0 + np.arange(len(grid)), index=grid)
        prime(materializer, 'a1', mirror, 'M', 'model')

        # Source aveugle : a1 n'y porte plus aucune observation
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=blind,
        )

        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'model'}
        # Le miroir, cellule par cellule, alors que la source est vide
        np.testing.assert_allclose(features['a1'].to_numpy(), mirror.to_numpy())

    def test_rank_two_origin_follows_store(self, reference_timeseries):
        """Rank 2: the origin is READ, never inferred from the way."""
        grid = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        mirror = pd.Series(7.0 + np.arange(len(grid)), index=grid)
        # Étape produite par repli : le miroir porte 'interpolated'
        prime(materializer, 'a1', mirror, 'M', 'interpolated')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # La voie reste celle du rang 2, seule l'origine change
        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'interpolated'}
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}
        np.testing.assert_allclose(features['a1'].to_numpy(), mirror.to_numpy())

    @pytest.mark.parametrize('origin', ['model', 'interpolated', 'observed'])
    def test_rank_two_trigger_ignores_the_origin(self, origin):
        """Rank 2 fires on the mere presence of cells produced at ``f``.

        « Le repli matérialise » (§4.4) : une étape servie par repli est
        relue comme une autre. Le code et la docstring de ``classify`` le
        disent ; la docstring de classe, qui exige ``origin == 'model'``, non
        (ANO-FREQ-041, [DOC]).
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', origin)
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'stage_model'

    def test_rank_three_carries_from_previous_stage(self, reference_timeseries):
        """Rank 3: the quarterly imputation of a1 is CARRIED onto the M grid.

        Answer to the first cause of B28: a1 is no longer left NaN two months
        out of three at the monthly stage.
        """
        quarterly = pd.Series(20.0 + np.arange(12), index=quarterly_grid())
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', quarterly, 'Q', 'model')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=True,
        )

        assert ways == {'a1': 'carried_model'}
        # Aucune cellule NaN : c'est exactement le défaut corrigé
        assert features['a1'].notna().all()
        # L'interpolation d'une valeur de modèle RESTE de modèle
        assert origins == {'a1': 'model'}
        assert set(materializer.origin_store['a1'].unique()) == {'model'}
        # Les cellules reportées sont écrites à la fréquence de l'étape courante
        assert set(materializer.imputed_freq_store['a1'].unique()) == {'M'}

    def test_rank_three_rescales_to_origin_stage_totals(self, reference_timeseries):
        """Rank 3: the rescaling bears on the totals of ``f'``, not those of ``f``."""
        quarterly_dates = quarterly_grid()
        applier = SpyApplier()
        materializer = CovariateMaterializer(
            covariate_strategy='model', aggregation_constraint_applier=applier
        )
        quarterly = pd.Series(20.0 + np.arange(12), index=quarterly_dates)
        prime(materializer, 'a1', quarterly, 'Q', 'model')

        _features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'carried_model'}
        assert len(applier.calls) == 1
        call = applier.calls[0]
        # Les totaux de l'étape d'origine, jamais ceux de l'étape courante ;
        # la colonne, pour une contrainte par colonne ; la fréquence de la
        # grille CIBLE, pour la garde de complétude calendaire
        assert (call['period_freq'], call['column'], call['grid_freq']) == ('Q', 'a1', 'M')
        # Ni les ancres annuelles de la source, ni la grille mensuelle
        pd.testing.assert_series_equal(call['observations'], quarterly, check_names=False)

    def test_rank_four_fallback_interpolate(self, reference_timeseries):
        """Rank 4, empty stores: ``covariate_fallback='interpolate'``."""
        materializer = CovariateMaterializer(
            covariate_strategy='model', covariate_fallback='interpolate'
        )

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'interpolate'}
        assert origins == {'a1': 'interpolated'}
        assert features['a1'].notna().all()

    def test_rank_four_fallback_tolerate_nan(self, reference_timeseries):
        """Rank 4, empty stores: ``covariate_fallback='tolerate_nan'``."""
        materializer = CovariateMaterializer(
            covariate_strategy='model', covariate_fallback='tolerate_nan'
        )

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'raw_anchors'}
        assert origins == {'a1': 'observed'}
        # Les trois ancres annuelles, et NaN partout ailleurs
        assert features['a1'].notna().sum() == 3

    def test_precedence_stops_at_first_applicable_rank(self, reference_timeseries):
        """Present at ``f`` AND at an earlier stage, the covariate takes rank 2.

        D37 as well: a value produced at the coarser stage never lands, raw,
        on the finer grid — the 2021 quarterly cells stay out of it.
        """
        grid = monthly_grid()
        quarters = quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')

        # Miroir mixte : 2021 imputée à Q, 2022-2023 imputées à M
        prime(materializer, 'a1', pd.Series(99.0, index=quarters[quarters.year == 2021]), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(1.0, index=grid[grid.year >= 2022]), 'M', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # Arrêt au rang 2 : aucun report du miroir trimestriel de 2021
        assert ways == {'a1': 'stage_model'}
        assert set(features['a1'].loc['2022'].unique()) == {1.0}
        assert features['a1'].loc['2021'].isna().all()

    def test_model_way_degrades_per_entity(self, mixed_freq_panel_heterogeneous):
        """The way is a column property; its applicability is measured per entity."""
        data = mixed_freq_panel_heterogeneous
        dates = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')

        # a1 imputée au pas M pour la seule entité FR
        french = pd.MultiIndex.from_product([['FR'], dates], names=data.index.names)
        prime(materializer, 'a1', pd.Series(7.0 + np.arange(len(dates)), index=french), 'M', 'model')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=panel_grid(data, dates),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            record=True,
        )

        # Une seule voie pour la colonne, la plus dégradée des verdicts
        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'model'}
        # FR lit le miroir ; DE et IT dégradent au rang 4 (repli interpolé)
        np.testing.assert_allclose(
            features.loc['FR', 'a1'].to_numpy(), 7.0 + np.arange(len(dates))
        )
        assert features.loc['DE', 'a1'].iloc[0] == pytest.approx(120.0)
        assert features['a1'].notna().all()
        # Chaque entité porte l'origine de la voie qui l'a effectivement servie
        assert set(materializer.origin_store['a1'].unique()) == {'model', 'interpolated'}

    @pytest.mark.parametrize(
        'strategy, expected',
        [('tolerate_nan', 'raw_anchors'), ('interpolate', 'interpolate')],
        ids=['tolerate_nan', 'interpolate'],
    )
    def test_ranks_two_and_three_unreachable_outside_model(self, strategy, expected):
        """§8.1: outside ``'model'``, no strategy ever reads the stores."""
        grid = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy=strategy)
        # Registres amorcés aux deux rangs : rien ne doit les lire
        prime(materializer, 'a1', pd.Series(20.0 + np.arange(12), index=quarterly_grid()), 'Q', 'model')
        prime(materializer, 'a2', pd.Series(7.0 + np.arange(len(grid)), index=grid), 'M', 'model')

        ways = materializer.decide_ways(
            columns=['m1', 'q1', 'a1', 'a2'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
        )
        assert ways == {'m1': 'identity', 'q1': expected, 'a1': expected, 'a2': expected}


class TestTwoCandidateRanks:
    """§4.4: two ranks applicable at once — only the higher one applies."""

    def test_rank_one_identity_beats_the_stage_mirror(self, reference_timeseries):
        """q1 observed at the stage frequency is read, never its mirror at ``f``.

        Valeurs d'or : q1 vaut 10 · k aux 12 fins de trimestre (§2.2).
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'q1', pd.Series(999.0, index=quarterly_grid()), 'Q', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['q1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        np.testing.assert_allclose(features['q1'].to_numpy(), 10.0 * np.arange(1, 13))

    def test_rank_one_aggregate_beats_the_stage_mirror(self, reference_timeseries):
        """m1, finer than the stage, is summed exactly, never read in the mirror.

        Valeur d'or : m1 vaut 100 + rang ; le trimestre k (base 0) somme
        3 · 100 + 9k + 3 = 303 + 9k.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'm1', pd.Series(999.0, index=quarterly_grid()), 'Q', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['m1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        np.testing.assert_allclose(features['m1'].to_numpy(), 303.0 + 9.0 * np.arange(12))

    def test_rank_one_beats_a_carry(self, reference_timeseries):
        """q1 imputed at an earlier (annual) stage is still read at its observations."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'q1', pd.Series(999.0, index=annual_grid()), 'Y', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['q1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert (ways, features['q1'].tolist()) == ({'q1': 'identity'}, list(10.0 * np.arange(1, 13)))

    def test_rank_two_beats_the_fallback(self, reference_timeseries):
        """The mirror at ``f`` wins over the interpolation of the anchors 120 / 132 / 150."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert (ways, set(features['a1'])) == ({'a1': 'stage_model'}, {7.0})

    def test_rank_three_beats_the_fallback(self, reference_timeseries):
        """A carry of the constant quarterly imputation 5.0 is constant, never the anchors."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(5.0, index=quarterly_grid()), 'Q', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert (ways, set(features['a1'].round(12))) == ({'a1': 'carried_model'}, {5.0})

    def test_carry_starts_from_the_closest_earlier_stage(self, reference_timeseries):
        """Mirror cells at Q and at Y: the carry onto M starts from Q.

        Cellules Q en 2021 (5.0) puis cellules Y en 2022-2023 (1.0) : l'étape
        antérieure la plus proche de M est Q, aucune valeur annuelle ne doit
        transparaître.
        """
        quarters = quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(5.0, index=quarters[quarters.year == 2021]), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(1.0, index=annual_grid()[1:]), 'Y', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert set(features['a1'].dropna().round(12)) == {5.0}

    def test_mirror_cell_without_production_frequency_grants_no_model_rank(self):
        """A mirror cell of unknown production frequency is neither rank 2 nor rank 3."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), None, 'model')
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'interpolate'

    def test_unrecognized_production_label_in_the_mirror_raises(self):
        """An unreadable production frequency is an error, never a silent rank."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=quarterly_grid()), 'zz', 'model')
        with pytest.raises(ValueError, match='Unsupported frequency: zz'):
            materializer.classify('a1', 'M', TS_FREQUENCIES)


class TestRankOnePerEntity:
    """§4.6: ``_applicable_way`` tests rank 1 before reading the stores, per entity."""

    def test_entity_observing_at_the_stage_reads_its_observations(
        self, mixed_freq_panel_multifrequency
    ):
        """PANEL-F at M: IT observes v monthly and keeps 10 / 11 / 12.5, never 999.

        Le miroir porte 999 pour les trois entités au pas M ; la voie de la
        colonne est 'stage_model' (verdicts de FR et DE), qu'IT ramène à
        l'identité. Valeurs d'or du §2.5.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        for entity in ('FR', 'DE', 'IT'):
            prime(materializer, 'v', pd.Series(999.0, index=entity_grid([entity], monthly_grid())), 'M', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            record=False,
        )
        expected_it = [10.0] * 12 + [11.0] * 12 + [12.5] * 12
        assert (ways, features.loc['IT', 'v'].tolist()) == ({'v': 'stage_model'}, expected_it)

    @pytest.mark.parametrize(
        'entity, expected',
        [
            # IT, plus fine que l'étape : sommes trimestrielles exactes (§2.5)
            ('IT', [30.0] * 4 + [33.0] * 4 + [37.5] * 4),
            # DE, à la fréquence de l'étape : ses 12 ancres
            ('DE', [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0]),
            # FR, plus grossière : le miroir de l'étape
            ('FR', [999.0] * 12),
        ],
        ids=['IT-aggregate', 'DE-identity', 'FR-mirror'],
    )
    def test_quarterly_stage_applies_rank_one_before_the_mirror(
        self, mixed_freq_panel_multifrequency, entity, expected
    ):
        """PANEL-F at Q, mirror at 999 for every entity: each entity's own rank."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        for name in ('FR', 'DE', 'IT'):
            prime(materializer, 'v', pd.Series(999.0, index=entity_grid([name], quarterly_grid())), 'Q', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            record=False,
        )
        np.testing.assert_allclose(features.loc[entity, 'v'].to_numpy(), expected)


class TestModelWaysPerEntity:
    """§4.4 per entity: a model way is brought back to what the entity's mirror holds."""

    @pytest.mark.parametrize(
        'entity, expected',
        [
            # FR ne porte que des cellules de l'étape M : rang 2, ses valeurs
            ('FR', [1.0] * 36),
            # DE ne porte que des cellules Q : rang 3, report de la constante
            ('DE', [5.0] * 36),
        ],
        ids=['FR-stage-cells', 'DE-carry'],
    )
    def test_each_entity_is_served_by_the_rank_its_mirror_holds(
        self, mixed_freq_panel_heterogeneous, entity, expected
    ):
        """DE listed first makes the column way ``'carried_model'``; FR still reads its stage.

        Les valeurs servies ne dépendent que du miroir de chaque entité ; la
        voie de colonne, elle, dépend de l'ordre des entités (ANO-FREQ-038) et
        n'est donc pas assertée ici.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(1.0, index=entity_grid(['FR'], monthly_grid())), 'M', 'model')
        prime(materializer, 'a1', pd.Series(5.0, index=entity_grid(['DE'], quarterly_grid())), 'Q', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=entity_grid(['DE', 'FR'], monthly_grid()),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=mixed_freq_panel_heterogeneous,
            record=False,
        )
        np.testing.assert_allclose(features.loc[entity, 'a1'].to_numpy(), expected)

    def test_entity_without_frequency_under_a_model_way_stays_empty(self, mixed_freq_panel_heterogeneous):
        """IT never observes a1: under the column's model way, its rows stay NaN.

        La voie de colonne est 'stage_model' (miroir de FR) ; IT, sans
        fréquence détectée, n'a ni rang 1, ni miroir, ni conversion
        définissable : ses ancres brutes, vides.
        """
        data = mixed_freq_panel_heterogeneous.copy()
        data.loc['IT', 'a1'] = np.nan
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=entity_grid(['FR'], monthly_grid())), 'M', 'model')

        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
            stage_freq='M',
            detected_frequencies={'a1': {('FR',): 'Y', ('DE',): 'Y'}},
            source_data=data,
            record=False,
        )
        assert (ways, features.loc['IT', 'a1'].isna().all()) == ({'a1': 'stage_model'}, True)
