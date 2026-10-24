"""Tests of ``CovariateMaterializer.stage_frame``, the working view of one stage.

The stage frame carries the input values, the EXACT aggregations of the finer
columns onto the stage grid and the mirror of the earlier stages; it
fabricates no interpolated value — the cap is applied entity by entity, after
the precedence: any entity whose own way lands on the interpolation stays at
its raw anchors — and writes no store. The mirror is read under
``covariate_strategy='model'`` only (by precedence), filtered by production
frequency so that a value produced at a coarser stage never lands, raw, on a
finer grid.

An unbound entity of a unanimous stage binding is covered, and a finer entity
is aggregated exactly, never given point values — both reachable through the
stage step of ``HighFrequencyImputer``; the per-entity cap applies entity by
entity.

Covered symbol: ``CovariateMaterializer.stage_frame``.
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer

# Jeux et perturbations partagés
from tests.support.perturbations import shuffle_rows
from tests.unit.imputation._components.covariate_materializer.helpers import (
    PANEL_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    TS_FREQUENCIES,
    entity_grid,
    monthly_grid,
    panel_grid,
    prime,
    quarterly_grid,
)


class TestStageFrameContent:
    """Originals, exact aggregations and raw anchors — nothing interpolated, nothing written."""

    @pytest.mark.parametrize(
        'column, expected',
        [
            # m1, plus fine que l'étape : somme exacte 303 + 9k
            ('m1', 303.0 + 9.0 * np.arange(12)),
            # q1, à la fréquence de l'étape : ses 12 observations
            ('q1', 10.0 * np.arange(1, 13)),
            # a1, plus grossière : ses 3 ancres de fin d'année, NaN ailleurs
            ('a1', np.array([np.nan] * 3 + [120.0] + [np.nan] * 3 + [132.0] + [np.nan] * 3 + [150.0])),
        ],
        ids=['finer-aggregate', 'equal-identity', 'coarser-raw-anchors'],
    )
    def test_originals_exact_aggregations_and_raw_anchors(self, reference_timeseries, column, expected):
        """``TS`` at stage Q, default strategy: the three kinds of exact cells."""
        frame = CovariateMaterializer().stage_frame(
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        np.testing.assert_allclose(frame[column].to_numpy(), expected)

    @pytest.mark.parametrize(
        'strategy, fallback',
        [('tolerate_nan', 'interpolate'), ('interpolate', 'interpolate'),
         ('model', 'interpolate'), ('model', 'tolerate_nan')],
        ids=['tolerate_nan', 'interpolate', 'model-interpolate', 'model-tolerate_nan'],
    )
    def test_nothing_is_interpolated_whatever_the_strategy(self, reference_timeseries, strategy, fallback):
        """A rank-4 column stays at its 3 anchors on the monthly grid, under every strategy."""
        frame = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback).stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert int(frame['a1'].notna().sum()) == 3

    def test_columns_none_takes_every_source_column_in_order(self, reference_timeseries):
        """Without ``columns``, the frame carries the source columns, in their order."""
        frame = CovariateMaterializer().stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert list(frame.columns) == list(reference_timeseries.columns)

    def test_columns_subset_keeps_the_given_order(self, reference_timeseries):
        """An explicit ``columns`` fixes both the subset and the order."""
        frame = CovariateMaterializer().stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            columns=['a2', 'm1'],
        )
        assert list(frame.columns) == ['a2', 'm1']

    def test_nothing_is_written_to_the_stores(self, reference_timeseries):
        """The frame is a view: a primed mirror is read, never extended nor altered."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        mirror = pd.Series(7.0, index=monthly_grid())
        prime(materializer, 'a1', mirror, 'M', 'model')

        materializer.stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        # Seule la colonne amorcée est présente, telle qu'amorcée
        assert list(materializer.imputed_store) == ['a1']
        pd.testing.assert_series_equal(materializer.imputed_store['a1'], mirror)

    def test_all_nan_column_is_nan_without_error(self, reference_timeseries):
        """An empty column is carried as NaN, without error."""
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan
        frame = CovariateMaterializer().stage_frame(
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=blind,
        )
        assert frame['a1'].isna().all()


class TestStageFrameMirror:
    """The mirror is read under ``'model'`` only, filtered by production frequency."""

    @pytest.mark.parametrize(
        'strategy, expected',
        [('interpolate', 3), ('tolerate_nan', 3), ('model', 36)],
    )
    def test_mirror_is_read_under_model_only(self, reference_timeseries, strategy, expected):
        """A mirror of a1 at M fills the monthly grid under ``'model'``, nowhere else."""
        materializer = CovariateMaterializer(covariate_strategy=strategy)
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', 'model')

        frame = materializer.stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert int(frame['a1'].notna().sum()) == expected

    def test_coarser_production_never_lands_raw_on_a_finer_grid(self, reference_timeseries):
        """2021 imputed at Q, 2022-2023 at M: the M frame shows no quarterly value.

        The mirror holds nothing for 2021 at M: the frame keeps the raw
        observation of 2021 at its anchor — the mirror overlays the anchors
        without replacing them —, never the quarterly
        production (99).
        """
        grid, quarters = monthly_grid(), quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(99.0, index=quarters[quarters.year == 2021]), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(1.0, index=grid[grid.year >= 2022]), 'M', 'model')

        frame = materializer.stage_frame(
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        # Valeur d'or : seule l'observation annuelle de 2021, 120, à sa date
        assert frame['a1'].loc['2021'].dropna().to_dict() == {pd.Timestamp('2021-12-31'): 120.0}

    @pytest.mark.parametrize('fallback', ['interpolate', 'tolerate_nan'])
    def test_entity_with_an_empty_mirror_is_capped_to_its_raw_anchors(
        self, mixed_freq_panel_heterogeneous, fallback
    ):
        """A model way degraded to the fallback for one entity is capped like any other.

        FR porte un miroir de a1 au pas M, DE n'en a pas : la voie de colonne
        est 'stage_model', ramenée pour DE au repli. Le plafonnement se fait
        par entité, après la précédence : sous les deux replis, DE reste à ses
        3 ancres annuelles — la vue d'étape ne fabrique aucune interpolation.
        """
        data = mixed_freq_panel_heterogeneous
        materializer = CovariateMaterializer(covariate_strategy='model', covariate_fallback=fallback)
        prime(materializer, 'a1', pd.Series(7.0, index=entity_grid(['FR'], monthly_grid())), 'M', 'model')

        frame = materializer.stage_frame(
            grid_index=panel_grid(data, monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            columns=['a1'],
        )
        assert int(frame.loc['DE', 'a1'].notna().sum()) == 3

    def test_carried_mirror_is_not_capped(self, reference_timeseries):
        """Rank 3 reads the mirror: the cap leaves it, only an interpolation way is capped.

        Imputation trimestrielle constante (5.0), reportée sur la grille
        mensuelle : la vue d'étape la porte sur les 36 mois.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(5.0, index=quarterly_grid()), 'Q', 'model')

        frame = materializer.stage_frame(
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            columns=['a1'],
        )
        assert set(frame['a1'].round(12)) == {5.0}


class TestStageFramePanel:
    """Panels: per-entity frequencies, empty entity, partial stage bindings."""

    @pytest.mark.parametrize(
        'entity, expected',
        [
            # FR, annuelle, plus grossière que Q : ses 3 ancres de décembre
            ('FR', [np.nan] * 3 + [120.0] + [np.nan] * 3 + [132.0] + [np.nan] * 3 + [150.0]),
            # DE, trimestrielle : ses 12 ancres
            ('DE', [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0]),
        ],
        ids=['FR-raw-anchors', 'DE-identity'],
    )
    def test_heterogeneous_frequencies_at_the_quarterly_stage(
        self, mixed_freq_panel_multifrequency, entity, expected
    ):
        """PANEL-F ``v`` at stage Q: each entity at its own exact cells."""
        frame = CovariateMaterializer().stage_frame(
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
        )
        np.testing.assert_allclose(frame.loc[entity, 'v'].to_numpy(), expected)

    def test_finer_entity_is_aggregated_exactly(self, mixed_freq_panel_multifrequency):
        """PANEL-F ``v`` at stage Q: IT's months are summed — 30 / 33 / 37.5.

        The 'interpolate' column way is not capped as a whole: each entity
        follows its own way, and IT, finer than the stage, receives the exact
        aggregation — never the point value of the month (10 / 11 / 12.5).
        """
        frame = CovariateMaterializer().stage_frame(
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
        )
        np.testing.assert_allclose(
            frame.loc['IT', 'v'].to_numpy(), [30.0] * 4 + [33.0] * 4 + [37.5] * 4
        )

    def test_entity_without_the_column_stays_nan(self, mixed_freq_panel_heterogeneous):
        """climat_affaires: NaN for IT, observed for FR and DE."""
        data = mixed_freq_panel_heterogeneous
        frame = CovariateMaterializer().stage_frame(
            grid_index=panel_grid(data, monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        filled = frame['climat_affaires'].notna().groupby(level=0).sum().to_dict()
        assert filled == {'DE': 36, 'FR': 36, 'IT': 0}

    def test_unanimous_stage_binding_covers_an_unbound_grid_entity(self, mixed_freq_panel_heterogeneous):
        """The grid holds IT, the binding names FR and DE only.

        Golden value: m1 of IT summed over Q1 2021, 100 + 101 + 102 = 303.
        """
        frame = CovariateMaterializer().stage_frame(
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq={('FR',): 'Q', ('DE',): 'Q'},
            detected_frequencies=TS_FREQUENCIES,
            source_data=mixed_freq_panel_heterogeneous,
            columns=['m1'],
        )
        assert frame.loc['IT', 'm1'].iloc[0] == pytest.approx(303.0)


class TestStageFrameRobustness:
    """Row order and start-anchored grids."""

    def test_shuffled_source_gives_the_same_frame(self, reference_timeseries):
        """The source row order plays no part."""
        kwargs = dict(grid_index=quarterly_grid(), stage_freq='Q', detected_frequencies=TS_FREQUENCIES)
        reference = CovariateMaterializer().stage_frame(source_data=reference_timeseries, **kwargs)
        shuffled = CovariateMaterializer().stage_frame(
            source_data=shuffle_rows(reference_timeseries, seed=3), **kwargs
        )
        pd.testing.assert_frame_equal(shuffled, reference)

    def test_quarter_start_grid_on_a_month_start_source(self):
        """A month-start source sums onto a quarter-start grid: 303 + 9k."""
        dates = pd.date_range('2021-01-01', periods=36, freq='MS')
        data = pd.DataFrame({'m1': 100.0 + np.arange(36)}, index=dates)
        frame = CovariateMaterializer().stage_frame(
            grid_index=pd.date_range('2021-01-01', periods=12, freq='QS'),
            stage_freq='Q',
            detected_frequencies={'m1': 'M'},
            source_data=data,
        )
        np.testing.assert_allclose(frame['m1'].to_numpy(), 303.0 + 9.0 * np.arange(12))
