"""Tests of the three stores held by ``CovariateMaterializer``.

The mirror of the values produced (``imputed_store``), their production
frequency (``imputed_freq_store``) and their cell origin (``origin_store``)
are written together, NaN cells excluded, by ``new.combine_first(existing)``
(cells a production does not cover survive). "Fallback materializes": the
fallback interpolation feeds the stores like a model prediction. ``reset``
separates two passes — a transform never reads the fit's mirror — and
``snapshot`` returns an independent deep copy. Frequencies and origins are stored
in the ``str`` dtype, whatever the dtype of the series received.

Covered symbols: ``CovariateMaterializer.record_production``,
``interpolate_column``, ``snapshot``, ``reset``, ``imputed_store``,
``imputed_freq_store``, ``origin_store``.
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer
from tsforecast.imputation._components.provenance import CELL_ORIGIN_DTYPE

# Jeux et perturbations partagés
from tests.support.perturbations import shuffle_rows
from tests.unit.imputation._components.covariate_materializer.helpers import (
    PANEL_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    TS_FREQUENCIES,
    entity_grid,
    monthly_grid,
    origins_per_entity,
    panel_grid,
    prime,
    quarterly_grid,
)


def _record(materializer, column, values, freq='M', origin='model'):
    """Record ``values`` with one production frequency and one origin."""
    materializer.record_production(
        column,
        values,
        pd.Series(origin, index=values.index, dtype='str'),
        pd.Series(freq, index=values.index, dtype='str'),
    )


class TestRecordProduction:
    """The public writer of the stores, shared with ``materialize`` and ``interpolate_column``."""

    def test_feeds_the_three_stores_on_the_same_index(self):
        """NaN cells are left out of the three stores alike."""
        dates = monthly_grid(periods=3)
        materializer = CovariateMaterializer()
        _record(materializer, 'a1', pd.Series([1.0, np.nan, 3.0], index=dates))

        kept = dates[[0, 2]]
        assert [
            store['a1'].index.equals(kept)
            for store in (materializer.imputed_store, materializer.imputed_freq_store, materializer.origin_store)
        ] == [True, True, True]

    def test_new_production_wins_on_the_overlap_and_old_cells_survive(self):
        """B5: ``new.combine_first(existing)`` — overlap rewritten, the rest kept."""
        dates = monthly_grid(periods=2)
        materializer = CovariateMaterializer()
        _record(materializer, 'a1', pd.Series([1.0, 2.0], index=dates), origin='interpolated')
        _record(materializer, 'a1', pd.Series([9.0], index=dates[1:]), origin='model')

        # Valeur et origine réécrites ensemble sur la date commune
        assert (
            materializer.imputed_store['a1'].tolist(), materializer.origin_store['a1'].tolist(),
        ) == ([1.0, 9.0], ['interpolated', 'model'])

    def test_all_nan_production_writes_and_erases_nothing(self):
        """A production without any value leaves the stores exactly as they were."""
        dates = monthly_grid(periods=2)
        materializer = CovariateMaterializer()
        _record(materializer, 'a1', pd.Series([1.0, 2.0], index=dates))
        _record(materializer, 'a1', pd.Series(np.nan, index=dates))
        # Production vide sur une colonne jamais écrite : rien n'apparaît
        _record(materializer, 'a2', pd.Series(np.nan, index=dates))

        assert (materializer.imputed_store['a1'].tolist(), 'a2' in materializer.imputed_store) == (
            [1.0, 2.0], False,
        )

    def test_unsorted_production_is_read_back_aligned(self, reference_timeseries):
        """A production recorded in shuffled order is read by label at rank 2."""
        dates = monthly_grid()
        # Valeur = rang du mois, enregistrée dans le désordre
        production = shuffle_rows(pd.Series(np.arange(36, dtype=float), index=dates), seed=1)
        materializer = CovariateMaterializer(covariate_strategy='model')
        _record(materializer, 'a1', production)

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=dates,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=False,
        )
        np.testing.assert_allclose(features['a1'].to_numpy(), np.arange(36, dtype=float))

    @pytest.mark.parametrize('label', ['M', 'ME', 'MS', 'monthly'])
    def test_position_aliases_of_the_production_frequency_are_recognized(self, label):
        """A production frequency is compared by base: ``'ME'`` / ``'MS'`` are the stage ``'M'``."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        _record(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), freq=label)
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'stage_model'


class TestInterpolateColumn:
    """The interpolation of the class: strategy, ``covariate_fallback`` and failure fallback."""

    def test_fallback_writes_to_all_three_stores(self, reference_timeseries):
        """« Le repli matérialise »: the three stores are fed."""
        materializer = CovariateMaterializer()

        values = materializer.interpolate_column(
            'a1', monthly_grid(), 'M', TS_FREQUENCIES, reference_timeseries
        )

        # Les trois registres portent la colonne, sur le même index
        for store in (
            materializer.imputed_store,
            materializer.imputed_freq_store,
            materializer.origin_store,
        ):
            assert 'a1' in store
            assert store['a1'].index.equals(values.dropna().index)

        # Miroir des valeurs, fréquence de production et origine
        pd.testing.assert_series_equal(
            materializer.imputed_store['a1'], values.dropna(), check_names=False
        )
        assert set(materializer.imputed_freq_store['a1'].unique()) == {'M'}
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}

    def test_missing_column_raises_key_error(self, reference_timeseries):
        """A column absent from the source is an error."""
        with pytest.raises(KeyError, match='zz'):
            CovariateMaterializer().interpolate_column(
                'zz', monthly_grid(), 'M', TS_FREQUENCIES, reference_timeseries
            )

    def test_duplicated_source_dates_raise(self, reference_timeseries):
        """The fallback interpolation refuses duplicated dates like ``materialize``."""
        duplicated = pd.concat([reference_timeseries, reference_timeseries.iloc[-1:]])
        with pytest.raises(ValueError, match='Duplicate dates in source_data'):
            CovariateMaterializer().interpolate_column(
                'a1', monthly_grid(), 'M', TS_FREQUENCIES, duplicated
            )

    def test_entity_without_observation_is_left_out_of_the_stores(self, mixed_freq_panel_heterogeneous):
        """IT never observes a1 here: its rows are NaN and absent from the stores."""
        data = mixed_freq_panel_heterogeneous.copy()
        data.loc['IT', 'a1'] = np.nan
        materializer = CovariateMaterializer()

        materializer.interpolate_column(
            'a1', panel_grid(data, monthly_grid()), 'M', PANEL_FREQUENCIES, data
        )
        assert set(materializer.origin_store['a1'].index.get_level_values(0)) == {'DE', 'FR'}

    def test_each_entity_records_the_origin_of_its_own_way(self, mixed_freq_panel_multifrequency):
        """PANEL-F at M: FR and DE interpolated, IT brought back to its rank 1.

        On its own grid, the interpolation of a column IS the identity:
        IT, monthly, is read at its observations, with origin 'observed'.
        """
        materializer = CovariateMaterializer()
        materializer.interpolate_column(
            'v',
            entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
            'M',
            PANEL_F_FREQUENCIES,
            mixed_freq_panel_multifrequency,
        )
        assert origins_per_entity(materializer.origin_store['v']) == {
            'DE': {'interpolated'}, 'FR': {'interpolated'}, 'IT': {'observed'},
        }

    def test_all_nan_column_leaves_the_stores_untouched(self, reference_timeseries):
        """Nothing interpolated, nothing recorded."""
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan
        materializer = CovariateMaterializer()
        values = materializer.interpolate_column('a1', monthly_grid(), 'M', TS_FREQUENCIES, blind)
        assert (values.isna().all(), materializer.imputed_store) == (True, {})

    def test_shuffled_source_gives_the_same_values(self, reference_timeseries):
        """The source row order plays no part."""
        reference = CovariateMaterializer().interpolate_column(
            'a1', monthly_grid(), 'M', TS_FREQUENCIES, reference_timeseries
        )
        shuffled = CovariateMaterializer().interpolate_column(
            'a1', monthly_grid(), 'M', TS_FREQUENCIES, shuffle_rows(reference_timeseries, seed=4)
        )
        pd.testing.assert_series_equal(shuffled, reference)

    def test_month_start_grid_passes_through_the_january_anchors(self):
        """Start-anchored data: the values are kept at their January anchors."""
        dates = pd.date_range('2021-01-01', periods=36, freq='MS')
        data = pd.DataFrame({'a1': np.nan}, index=dates)
        data.loc[dates[dates.month == 1], 'a1'] = [120.0, 132.0, 150.0]

        values = CovariateMaterializer().interpolate_column('a1', dates, 'M', {'a1': 'Y'}, data)
        assert (values.notna().all(), values.loc[dates[dates.month == 1]].tolist()) == (
            True, [120.0, 132.0, 150.0],
        )


class TestCombineFirst:
    """B5 through ``materialize``: a write never destroys the cells it does not cover."""

    def test_combine_first_never_overwrites(self, reference_timeseries):
        """A second production on a disjoint grid lets the first one survive."""
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')

        materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid('2021-01-31', 12),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=True,
        )
        first_state = materializer.imputed_store['a1'].copy()
        assert first_state.index.tolist() == [pd.Timestamp('2021-12-31')]

        # Seconde écriture sur une grille disjointe
        materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid('2022-01-31', 12),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=True,
        )
        store = materializer.imputed_store['a1']

        # L'index a crû vers l'union : la première ancre a survécu
        assert store.index.tolist() == [pd.Timestamp('2021-12-31'), pd.Timestamp('2022-12-31')]
        assert store.loc['2021-12-31'] == pytest.approx(120.0)
        assert store.loc['2022-12-31'] == pytest.approx(132.0)
        # Les trois registres ont crû ensemble
        assert len(materializer.origin_store['a1']) == 2
        assert len(materializer.imputed_freq_store['a1']) == 2

    def test_origin_store_values(self, reference_timeseries):
        """The origin rules of the stores."""
        grid = quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='interpolate')

        materializer.materialize(
            columns=['q1', 'm1', 'a1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=True,
        )

        # 1. Une cellule d'entrée non NaN vaut 'observed'
        assert set(materializer.origin_store['q1'].unique()) == {'observed'}
        # 2. Une agrégation exacte vaut 'observed'
        assert set(materializer.origin_store['m1'].unique()) == {'observed'}
        # 3. Une interpolation vaut 'interpolated'
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}

        # 3 bis. Un repli d'interpolation vaut aussi 'interpolated'
        fallback = CovariateMaterializer()
        fallback.interpolate_column('a2', grid, 'Q', TS_FREQUENCIES, reference_timeseries)
        assert set(fallback.origin_store['a2'].unique()) == {'interpolated'}

        # 4. Une prédiction de modèle vaut 'model', relue au rang 2
        model = CovariateMaterializer(covariate_strategy='model')
        prime(model, 'a1', pd.Series(20.0 + np.arange(12), index=grid), 'Q', 'model')
        _features, _ways, origins = model.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization={'a1': 'stage_model'},
            record=True,
        )
        assert origins == {'a1': 'model'}
        assert set(model.origin_store['a1'].unique()) == {'model'}


class TestSnapshot:
    """``snapshot`` returns a deep copy, independent in both directions."""

    @staticmethod
    def _primed():
        """Build a materializer holding one recorded column."""
        materializer = CovariateMaterializer()
        _record(materializer, 'a1', pd.Series([1.0, 2.0], index=monthly_grid(periods=2)))
        return materializer

    def test_keys_of_a_fresh_snapshot(self):
        """Three keys, three empty dicts."""
        assert CovariateMaterializer().snapshot() == {'imputed': {}, 'imputed_freq': {}, 'origin': {}}

    def test_mutating_a_store_in_place_leaves_the_snapshot_intact(self):
        """An in-place write into a store never reaches an earlier snapshot."""
        materializer = self._primed()
        snapshot = materializer.snapshot()
        materializer.imputed_store['a1'].iloc[0] = -1.0
        materializer.origin_store['a1'].iloc[0] = 'observed'
        assert (snapshot['imputed']['a1'].tolist(), snapshot['origin']['a1'].tolist()) == (
            [1.0, 2.0], ['model', 'model'],
        )

    def test_mutating_the_snapshot_leaves_the_stores_intact(self):
        """An in-place write into a snapshot never reaches the stores."""
        materializer = self._primed()
        snapshot = materializer.snapshot()
        snapshot['imputed']['a1'].iloc[0] = -1.0
        snapshot['imputed_freq'].clear()
        assert (materializer.imputed_store['a1'].tolist(), list(materializer.imputed_freq_store)) == (
            [1.0, 2.0], ['a1'],
        )

    def test_panel_series_are_copied_whole(self, mixed_freq_panel_heterogeneous):
        """A panel-indexed store is copied with its ``MultiIndex``."""
        materializer = CovariateMaterializer()
        grid = panel_grid(mixed_freq_panel_heterogeneous, monthly_grid())
        materializer.interpolate_column('a1', grid, 'M', PANEL_FREQUENCIES, mixed_freq_panel_heterogeneous)
        snapshot = materializer.snapshot()
        pd.testing.assert_series_equal(snapshot['origin']['a1'], materializer.origin_store['a1'])


class TestReset:
    """``reset`` separates two successive uses of one component."""

    def test_reset_empties_the_three_stores(self, reference_timeseries):
        """After ``reset``, the three stores are empty dicts."""
        materializer = CovariateMaterializer()
        materializer.interpolate_column('a1', monthly_grid(), 'M', TS_FREQUENCIES, reference_timeseries)
        materializer.reset()
        assert (
            materializer.imputed_store, materializer.imputed_freq_store, materializer.origin_store,
        ) == ({}, {}, {})

    def test_references_taken_before_reset_are_untouched(self, reference_timeseries):
        """``reset`` rebinds fresh dicts: a reference held elsewhere keeps its content."""
        materializer = CovariateMaterializer()
        materializer.interpolate_column('a1', monthly_grid(), 'M', TS_FREQUENCIES, reference_timeseries)
        held = materializer.origin_store
        materializer.reset()
        assert (list(held), materializer.origin_store is held) == (['a1'], False)

    def test_transform_after_reset_never_reads_the_fit_mirror(self):
        """The fit's imputation of a1 at M does not grant rank 2 after ``reset``."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', 'model')
        materializer.reset()
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'interpolate'

    def test_without_reset_a_second_use_reads_the_first_mirror(self):
        """Counter-test: without ``reset``, the second pass sees the first one's rank 2."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(7.0, index=monthly_grid()), 'M', 'model')
        assert materializer.classify('a1', 'M', TS_FREQUENCIES) == 'stage_model'

    def test_two_successive_uses_are_isolated(self, mixed_freq_panel_heterogeneous, reference_timeseries):
        """Use 1 (PANEL), snapshot, reset, use 2 (TS): use 2 behaves like a fresh component.

        Usage 1 : imputation de a1 au pas M sur le panel, enregistrée ; usage
        2 : matérialisation de TS. Après reset, l'usage 2 rend exactement ce
        que rendrait une instance neuve, ses registres sont ceux de l'instance
        neuve, et l'instantané de l'usage 1 reste intact.
        """
        panel = mixed_freq_panel_heterogeneous
        reused = CovariateMaterializer(covariate_strategy='model')
        reused.interpolate_column('a1', panel_grid(panel, monthly_grid()), 'M', PANEL_FREQUENCIES, panel)
        first_use = reused.snapshot()
        reused.reset()

        fresh = CovariateMaterializer(covariate_strategy='model')
        kwargs = dict(
            columns=['m1', 'a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        second = reused.materialize(**kwargs)
        expected = fresh.materialize(**kwargs)

        pd.testing.assert_frame_equal(second[0], expected[0])
        assert (second[1:], sorted(reused.imputed_store)) == (expected[1:], sorted(fresh.imputed_store))
        # L'instantané du premier usage n'a pas été touché par le second
        assert first_use['origin']['a1'].index.nlevels == 2


class TestStoreDtypes:
    """Frequencies are stored in the ``str`` dtype and origins in ``CELL_ORIGIN_DTYPE``, whatever the dtype received."""

    @pytest.mark.parametrize('text_dtype', [object, 'str'])
    def test_stores_are_typed(self, text_dtype):
        """``object`` or ``str`` input, a missing origin included: typed stores and snapshot."""
        grid = monthly_grid(periods=3)
        materializer = CovariateMaterializer()

        materializer.record_production(
            'a1', pd.Series([1.0, 2.0, 3.0], index=grid),
            pd.Series(['model', 'interpolated', None], index=grid, dtype=text_dtype),
            pd.Series(['M', 'M', 'M'], index=grid, dtype=text_dtype),
        )

        assert materializer.origin_store['a1'].dtype == CELL_ORIGIN_DTYPE
        assert materializer.imputed_freq_store['a1'].dtype == 'str'
        assert materializer.origin_store['a1'].tolist()[:2] == ['model', 'interpolated']
        snapshot = materializer.snapshot()
        assert snapshot['origin']['a1'].dtype == CELL_ORIGIN_DTYPE and snapshot['imputed_freq']['a1'].dtype == 'str'

    def test_successive_productions_stay_typed(self):
        """A second production combined with the first keeps the dtypes of the stores."""
        grid = monthly_grid(periods=4)
        materializer = CovariateMaterializer()
        prime(materializer, 'a1', pd.Series([1.0, 2.0], index=grid[:2]), 'Q', 'model')
        prime(materializer, 'a1', pd.Series([3.0, 4.0], index=grid[2:]), 'M', 'interpolated')

        assert materializer.imputed_freq_store['a1'].tolist() == ['Q', 'Q', 'M', 'M']
        assert materializer.origin_store['a1'].dtype == CELL_ORIGIN_DTYPE
        assert materializer.imputed_freq_store['a1'].dtype == 'str'

    @pytest.mark.internal
    def test_combine_first_of_two_dtypes(self):
        """A production of another dtype covering the store replaces it without concatenating a remainder."""
        from tsforecast.imputation._components.covariate_materializer import _combine_first

        grid = monthly_grid(periods=3)
        produced = pd.Series(['model', 'model'], index=grid[:2], dtype='str')

        covered = _combine_first(produced, pd.Series([1.0], index=grid[1:2]))
        partial = _combine_first(produced, pd.Series([1.0, 2.0], index=grid[1:]))

        assert covered.dtype == 'str' and covered.tolist() == ['model', 'model']
        assert partial.tolist() == ['model', 'model', 2.0]
