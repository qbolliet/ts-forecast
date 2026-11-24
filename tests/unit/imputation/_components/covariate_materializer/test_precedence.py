"""Tests of the three-rank materialization precedence.

When a covariate must
be materialized on the grid of stage ``f``, the precedence applies in order
and stops at the FIRST applicable rank — rank 1 identity / exact aggregation,
rank 2 the covariate's own production at ``f`` (or at a finer frequency,
exactly aggregated onto ``f``) read from the mirror, rank 3
``covariate_fallback`` from the original anchors. Rank 2 reads the stores,
hence exists under ``covariate_strategy='model'`` only; a production of an
earlier, coarser stage is never carried onto a finer grid.

Each rank is first pinned on its own, then opposed to a lower rank that is
ALSO applicable, so that only the precedence decides — including rank 1
against the mirror, entity by entity (an imputation never
overwrites a measurement).

Covered symbols: ``CovariateMaterializer.classify``, ``materialize`` (model
way), ``record_production`` (mirror priming).
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer

# Jeux partagés
from tests.unit.imputation._components.covariate_materializer.helpers import (
    PANEL_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    TS_FREQUENCIES,
    annual_grid,
    entity_grid,
    monthly_grid,
    panel_grid,
    prime,
    quarterly_grid,
)


def fallback_values(source_data, grid, column='a1', stage_freq='M', frequencies=TS_FREQUENCIES):
    """Values the fallback serves with empty stores, the reference of rank 3."""
    features, _ways, _origins = CovariateMaterializer(covariate_strategy='model').materialize(
        columns=[column],
        grid_index=grid,
        stage_freq=stage_freq,
        detected_frequencies=frequencies,
        source_data=source_data,
    )
    return features[column]


class TestPrecedenceRanks:
    """The three ranks, each on its own."""

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

        "Fallback materializes": a stage served by the fallback is read back like
        any other.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', origin)
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'stage_model'

    def test_rank_two_aggregates_a_finer_production_onto_a_coarser_grid(self, reference_timeseries):
        """A monthly production read on a quarterly grid is summed over each complete quarter.

        Valeurs d'or : le mois k (base 0) vaut k ; le trimestre q somme
        3 · (3q) + 0 + 1 + 2 = 9q + 3.
        """
        months = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(np.arange(len(months), dtype=float), index=months), 'M', 'model')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert (ways, origins) == ({'a1': 'stage_model'}, {'a1': 'model'})
        np.testing.assert_allclose(features['a1'].to_numpy(), 9.0 * np.arange(12) + 3.0)

    def test_incomplete_period_of_the_finer_production_stays_empty(self, reference_timeseries):
        """A quarter the monthly production covers partially is never summed."""
        months = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        # Production à partir de février 2021 : le premier trimestre est incomplet
        prime(materializer, 'a1', pd.Series(1.0, index=months[1:]), 'M', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert np.isnan(features['a1'].iloc[0])
        assert set(features['a1'].iloc[1:]) == {3.0}

    def test_own_frequency_cells_fill_what_the_finer_production_misses(self, reference_timeseries):
        """Q cells of 2021, M cells from 2022: the quarterly grid reads each where it exists."""
        months = monthly_grid()
        quarters = quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(99.0, index=quarters[quarters.year == 2021]), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(1.0, index=months[months.year >= 2022]), 'M', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=quarters,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert set(features['a1'].loc['2021']) == {99.0}
        assert set(features['a1'].loc['2022':]) == {3.0}

    def test_earlier_stage_production_is_never_carried(self, reference_timeseries):
        """A quarterly imputation of a1 never reaches the M grid: rank 3 from the anchors.

        A model trained on a quarterly production would predict on its
        interpolation; the monthly grid is served what it would be served with
        empty stores.
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
        )

        assert (ways, origins) == ({'a1': 'interpolate'}, {'a1': 'interpolated'})
        pd.testing.assert_series_equal(
            features['a1'], fallback_values(reference_timeseries, monthly_grid())
        )

    def test_rank_three_fallback_interpolate(self, reference_timeseries):
        """Rank 3, empty stores: ``covariate_fallback='interpolate'``."""
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

    def test_rank_three_fallback_tolerate_nan(self, reference_timeseries):
        """Rank 3, empty stores: ``covariate_fallback='tolerate_nan'``."""
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

        A value produced at the coarser stage never lands, raw,
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
        # FR lit le miroir ; DE et IT dégradent au rang 3 (repli interpolé,
        # un douzième de l'ancre annuelle par mois)
        np.testing.assert_allclose(
            features.loc['FR', 'a1'].to_numpy(), 7.0 + np.arange(len(dates))
        )
        assert features.loc['DE', 'a1'].iloc[0] == pytest.approx(120.0 / 12)
        assert features['a1'].notna().all()
        # Chaque entité porte l'origine de la voie qui l'a effectivement servie
        assert set(materializer.origin_store['a1'].unique()) == {'model', 'interpolated'}

    @pytest.mark.parametrize(
        'strategy, expected',
        [('tolerate_nan', 'raw_anchors'), ('interpolate', 'interpolate')],
        ids=['tolerate_nan', 'interpolate'],
    )
    def test_rank_two_unreachable_outside_model(self, strategy, expected):
        """Outside ``'model'``, no strategy ever reads the stores."""
        grid = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy=strategy)
        # Registres amorcés : rien ne doit les lire
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
    """Two ranks applicable at once — only the higher one applies."""

    def test_rank_one_identity_beats_the_stage_mirror(self, reference_timeseries):
        """q1 observed at the stage frequency is read, never its mirror at ``f``.

        Valeurs d'or : q1 vaut 10 · k aux 12 fins de trimestre.
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

    def test_rank_one_beats_an_earlier_stage_production(self, reference_timeseries):
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

    def test_coarser_grid_reads_the_latest_production_aggregated(self, reference_timeseries):
        """Q production then M production: a Q grid reads the M one, summed over each quarter.

        Cas d'une ligne d'entraînement plus grossière que l'étape : la
        covariable déjà imputée au mensuel y est lue agrégée, jamais à la
        valeur que l'étape trimestrielle avait produite.
        """
        quarters = quarterly_grid()
        months = monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(10.0 + np.arange(len(quarters)), index=quarters), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(-1.0, index=months), 'M', 'model')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=quarters,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert (ways, origins) == ({'a1': 'stage_model'}, {'a1': 'model'})
        assert set(features['a1']) == {-3.0}

    def test_mirror_cell_without_production_frequency_grants_no_model_rank(self):
        """A mirror cell of unknown production frequency grants no rank 2."""
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
    """``_applicable_way`` tests rank 1 before reading the stores, per entity."""

    def test_entity_observing_at_the_stage_reads_its_observations(
        self, mixed_freq_panel_multifrequency
    ):
        """PANEL-F at M: IT observes v monthly and keeps 10 / 11 / 12.5, never 999.

        The mirror carries 999 for the three entities at step M; the column way is
        'stage_model' (verdicts of FR and DE), which IT brings back to identity.
        Golden values.
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
            # IT, plus fine que l'étape : sommes trimestrielles exactes
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
    """Per entity, the model way is brought back to what the entity's mirror holds."""

    def test_each_entity_is_served_by_the_rank_its_mirror_holds(self, mixed_freq_panel_heterogeneous):
        """FR holds stage cells and reads them; DE holds quarterly cells only and takes rank 3.

        The quarterly production of DE belongs to an earlier stage: it is never
        carried onto the monthly grid, DE is served its fallback as with empty
        stores.
        """
        data = mixed_freq_panel_heterogeneous
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(1.0, index=entity_grid(['FR'], monthly_grid())), 'M', 'model')
        prime(materializer, 'a1', pd.Series(5.0, index=entity_grid(['DE'], quarterly_grid())), 'Q', 'model')

        grid = entity_grid(['DE', 'FR'], monthly_grid())
        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
            record=False,
        )
        assert ways == {'a1': 'stage_model'}
        np.testing.assert_allclose(features.loc['FR', 'a1'].to_numpy(), [1.0] * 36)
        reference = fallback_values(data, grid)
        np.testing.assert_allclose(features.loc['DE', 'a1'].to_numpy(), reference.loc['DE'].to_numpy())

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
