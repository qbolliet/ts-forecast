"""Guards and panel behaviour of the rescaling.

Scenarios: degenerate periods and totals, one block per (entity, period) on a panel.
"""

# Modules de base
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.aggregation_constraint import (
    AggregationConstraint,
    DEFAULT_CONSTRAINT_KEY,
)

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
)

# Outils de test partagés du paquet
from tests.unit.frequency.aggregation_constraint.helpers import (
    _EXACT,
    _GRID_ALIASES,
    _RAW_2021,
    _annual_observations,
    _monthly_grid,
    _raw_predictions,
    _record_warnings,
    _to_panel,
)


# =============================================================================
# Recalage : gardes et totaux dégénérés
# =============================================================================
class TestRescaleGuards:
    """Guards against degenerate periods and totals."""

    def test_nan_sub_period_leaves_the_period_raw(self):
        """One NaN sub-period forbids the constraint: raw predictions kept, NaN kept."""
        values = _raw_predictions()
        values.iloc[3] = np.nan

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y'
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_nan_sub_period_does_not_block_the_other_periods(self):
        """Only the period holding the NaN escapes; the mask matches the moved cells."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(_RAW_2021 * 2, index=grid, name='a1')
        values.loc['2022-05-31'] = np.nan
        observations = pd.Series(np.nan, index=grid, name='a1')
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Cellules effectivement modifiées (NaN exclu des deux côtés)
        changed = (rescaled != values) & values.notna()
        pd.testing.assert_series_equal(mask, changed, check_names=False)
        assert mask.loc['2021'].all() and not mask.loc['2022'].any()

    def test_period_without_observation_is_left_raw(self):
        """A delayed series end has no total to impose."""
        values = _raw_predictions()
        observations = pd.Series(np.nan, index=values.index, name='a1')

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_observation_outside_the_grid_imposes_nothing(self):
        """A total whose period has no row on the grid is ignored."""
        values = _raw_predictions(2021)
        observations = pd.Series([132.0], index=[pd.Timestamp('2022-12-31')])

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    @pytest.mark.parametrize('position', ['start', 'end'])
    def test_periods_truncated_at_both_edges_are_left_raw(self, position):
        """Only the whole middle year is rescaled; the truncated edge years keep raw values.

        Grid March 2020 - October 2022: 2020 holds ten months, 2022 ten
        months, 2021 twelve. Spreading an annual total over ten months would
        over-attribute 20 % to each.
        """
        alias = _GRID_ALIASES[('M', position)]
        grid = pd.date_range('2020-03-01', '2022-10-31', freq=alias)
        values = pd.Series(1.0, index=grid)
        # Ancres de chaque année, au début ou à la fin, même hors grille
        anchor_dates = pd.date_range('2020-01-01', periods=3, freq=_GRID_ALIASES[('Y', position)])
        observations = pd.Series([100.0, 120.0, 140.0], index=anchor_dates)

        # Les deux années tronquées sont nommées par un avertissement unique
        with pytest.warns(UserWarning, match=r'not enforced.*2 observed period\(s\)'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq=alias
            )

        # Valeur d'or : 2021 → 120 / 12 = 10 par mois ; 2020 et 2022 intactes
        expected = pd.Series(1.0, index=grid)
        expected.loc['2021'] = 10.0
        pd.testing.assert_series_equal(rescaled, expected)
        assert mask.loc['2021'].all() and not mask.loc['2020'].any() and not mask.loc['2022'].any()

    def test_truncation_guard_is_opt_in(self):
        """Without grid_freq, a truncated period still receives its whole total.

        The count of expected sub-periods cannot be read off the values: the
        missing ones are ABSENT from the grid, not empty.
        """
        grid = pd.date_range('2021-03-31', periods=10, freq='ME')
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(120.0, index=[grid[-1]])

        rescaled, _mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        assert rescaled.sum() == pytest.approx(120.0, abs=_EXACT)

    def test_quarter_truncated_on_a_monthly_grid(self, reference_timeseries):
        """The guard holds for every frequency pair, not only the year."""
        # Deux mois pour un trimestre de trois
        grid = reference_timeseries.loc['2021-01-31':'2021-02-28'].index
        values = pd.Series([5.0, 5.0], index=grid, name='q1')
        observations = pd.Series([np.nan, 10.0], index=grid, name='q1')

        with pytest.warns(UserWarning, match='not enforced'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Q', grid_freq='M'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_zero_total_zeroes_every_sub_period(self):
        """A zero observed total is imposed: ratio 0, sum exactly 0, no warning."""
        values = _raw_predictions()

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=0.0), 'Y',
        )

        assert rescaled.tolist() == [0.0] * 12
        assert mask.all() and messages == []

    def test_all_zero_sub_periods_against_a_non_zero_total(self):
        """All-zero predictions: no division by zero, raw values kept, one warning."""
        values = _raw_predictions(values=[0.0] * 12)

        with warnings.catch_warnings():
            # Division par zéro proscrite, y compris sous forme de RuntimeWarning
            warnings.simplefilter('error', RuntimeWarning)
            with pytest.warns(UserWarning, match='aggregate to zero for 1 period'):
                rescaled, mask = AggregationConstraint('sum').rescale(
                    values, _annual_observations(), 'Y'
                )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_cancelling_sub_periods_against_a_non_zero_total(self):
        """Predictions summing exactly to zero are left raw, with the zero warning."""
        values = _raw_predictions(values=[1.0, -1.0] * 6)

        with pytest.warns(UserWarning, match='aggregate to zero'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, _annual_observations(), 'Y'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_all_zero_sub_periods_against_a_zero_total(self):
        """0 against 0: the constraint already holds, nothing moves, no warning."""
        values = _raw_predictions(values=[0.0] * 12)

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=0.0), 'Y',
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any() and messages == []

    def test_negative_total_with_negative_predictions(self):
        """A negative total with same-sign predictions is rescaled silently and exactly."""
        values = _raw_predictions(values=[-value for value in _RAW_2021])

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=-120.0), 'Y',
        )

        # Valeur d'or : même ratio que l'exemple de référence, 120 / 112.5, signes conservés
        assert rescaled.sum() == pytest.approx(-120.0, abs=_EXACT)
        assert rescaled.loc['2021-12-31'] == pytest.approx(-11.2)
        assert mask.all() and messages == []

    def test_negative_total_with_positive_predictions_flips_the_profile(self):
        """Opposite signs: the constraint prevails, every sub-period changes sign."""
        values = _raw_predictions()

        with pytest.warns(UserWarning, match='flipped the predicted profile'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, _annual_observations(total=-120.0), 'Y'
            )

        assert rescaled.sum() == pytest.approx(-120.0, abs=_EXACT)
        assert (rescaled < 0).all() and mask.all()

    def test_numerically_zero_aggregate_is_zero(self):
        """A float sum that is zero up to rounding takes the zero guard.

        0.1 + 0.2 - 0.3 is mathematically zero but evaluates to 5.55e-17: the
        ratio 12 / 5.55e-17 would multiply the predictions by 2e17 and the
        rescaled sum would come back as 16, not 12. Below the rounding bound of the
        summation (n * eps * sum|x| = 3 * 2.2e-16 * 0.6, about 4e-16), the
        aggregate counts as zero: raw values kept, zero warning.
        """
        grid = pd.date_range('2021-01-31', periods=3, freq='ME')
        values = pd.Series([0.1, 0.2, -0.3], index=grid)
        observations = pd.Series([np.nan, np.nan, 12.0], index=grid)

        with pytest.warns(UserWarning, match='aggregate to zero'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Q', grid_freq='M'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_tiny_same_sign_predictions_are_still_rescaled(self):
        """The zero bound is relative: tiny predictions of one sign are not zero."""
        values = _raw_predictions(values=[1e-20] * 12)

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y'
        )

        # Valeur d'or : 120 / 12 = 10 par mois
        np.testing.assert_allclose(rescaled.to_numpy(), [10.0] * 12, rtol=1e-12)
        assert mask.all()

    def test_duplicated_observation_dates_are_rejected(self):
        """A duplicated anchor raises instead of doubling the observed total.

        The two copies of the December anchor are never summed: that would
        rescale the year to 240 instead of 120, silently.
        """
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(120.0, index=grid[[-1, -1]])

        with pytest.raises(
            ValueError, match=r"Duplicate dates in observations: \[Timestamp\('2021-12-31"
        ):
            AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='M')

    def test_duplicated_grid_dates_are_rejected(self):
        """A duplicated sub-period would count twice in the aggregate: it raises."""
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid.append(grid[:1]))

        with pytest.raises(ValueError, match='Duplicate dates in values'):
            AggregationConstraint('sum').rescale(values, _annual_observations(), 'Y')

    def test_duplicated_panel_pairs_are_rejected(self):
        """On a panel, a repeated (entity, date) pair raises; one date per entity does not."""
        values = _to_panel(_raw_predictions())
        repeated = pd.concat([values, values.iloc[:1]])

        with pytest.raises(ValueError, match=r"Duplicate dates in values: \[\('FR'"):
            AggregationConstraint('sum').rescale(
                repeated, _to_panel(_annual_observations()), 'Y'
            )

    def test_nat_rows_never_form_a_period(self):
        """A row dated NaT belongs to no period: neither rescaled nor used as a total.

        It forms no block of its own: its observation never becomes a
        "total" imposed on itself.
        """
        index = pd.DatetimeIndex(list(_monthly_grid()) + [pd.NaT])
        values = pd.Series([10.0] * 12 + [7.0], index=index)
        observations = pd.Series(np.nan, index=index)
        observations.iloc[11] = 240.0
        observations.iloc[12] = 5.0

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        # Valeur d'or : 2021 → 240 / 12 = 20 par mois ; la ligne NaT garde 7
        assert rescaled.tolist() == [20.0] * 12 + [7.0]
        assert mask.tolist() == [True] * 12 + [False]

    def test_constraint_none_leaves_raw_values(self):
        """Under None, raw predictions are untouched and the total is overwritten."""
        values = _raw_predictions()

        rescaled, mask = AggregationConstraint(None).rescale(
            values, _annual_observations(), 'Y'
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()
        # Le total observé n'est pas reconstitué : 112.5 et non 120
        assert rescaled.sum() == pytest.approx(112.5)

    def test_dict_form_selects_the_column(self):
        """rescale reads the column's own entry of the dict form."""
        constraint = AggregationConstraint({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'})
        values = _raw_predictions()

        rescaled_a1, _ = constraint.rescale(values, _annual_observations(), 'Y', column='a1')
        rescaled_q1, _ = constraint.rescale(values, _annual_observations(), 'Y', column='q1')

        assert rescaled_a1.sum() == pytest.approx(112.5)
        assert rescaled_q1.sum() == pytest.approx(120.0, abs=_EXACT)

    def test_empty_grid(self):
        """An empty grid comes back empty, with an empty mask."""
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        rescaled, mask = AggregationConstraint('sum').rescale(empty, empty, 'Y', grid_freq='M')

        assert rescaled.empty and mask.empty

    def test_single_row_grid_is_a_truncated_period(self):
        """A single month cannot carry an annual total once the grid is declared."""
        grid = pd.DatetimeIndex(['2021-12-31'])
        values = pd.Series([3.0], index=grid)
        observations = pd.Series([120.0], index=grid)

        with pytest.warns(UserWarning, match='not enforced'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='M'
            )

        assert rescaled.tolist() == [3.0] and not mask.any()


# =============================================================================
# Recalage : panel
# =============================================================================
class TestRescalePanel:
    """Rescaling on a panel: one block per (entity, period)."""

    def test_each_entity_is_rescaled_on_its_own_total(self):
        """On a panel, FR observes 120, DE 240, blocks never mix."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = 240.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        sums = rescaled.groupby(level='country').sum()
        assert sums.to_dict() == pytest.approx({'FR': 120.0, 'DE': 240.0}, abs=_EXACT)
        assert rescaled.loc[('DE', pd.Timestamp('2021-12-31'))] == pytest.approx(22.4)
        assert mask.all()

    def test_entities_never_share_a_rescaling_block(self):
        """A partly predicted entity does not prevent the other one from being rescaled."""
        values = _to_panel(_raw_predictions())
        values.loc[('DE', pd.Timestamp('2021-05-31'))] = np.nan
        observations = _to_panel(_annual_observations())

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        assert mask.loc['FR'].all() and not mask.loc['DE'].any()
        assert rescaled.loc['FR'].sum() == pytest.approx(120.0, abs=_EXACT)

    def test_entity_without_observation_is_left_raw(self):
        """An entity that never observes the column keeps free predictions."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc['DE'] = np.nan

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled.loc['DE'], values.loc['DE'])
        assert mask.loc['FR'].all() and not mask.loc['DE'].any()

    def test_source_frequency_differing_per_entity(self):
        """FR annual, DE quarterly: one call per source-frequency group, each exact.

        The period frequency is a per-call setting; ``HighFrequencyImputer``
        groups entities by detected source frequency and calls ``rescale``
        once per group. Grid frequencies may also differ per entity.
        """
        months = _monthly_grid()
        index = pd.MultiIndex.from_product([['FR', 'DE'], months], names=['country', 'date'])
        values = pd.Series(1.0, index=index)
        observations = pd.Series(np.nan, index=index)
        # FR : un total annuel ; DE : quatre totaux trimestriels
        observations.loc[('FR', months[-1])] = 120.0
        observations.loc[[('DE', months[i]) for i in (2, 5, 8, 11)]] = [3.0, 6.0, 9.0, 12.0]

        groups = {'Y': ['FR'], 'Q': ['DE']}
        pieces = []
        for period_freq, entities in groups.items():
            rows = index.get_level_values('country').isin(entities)
            rescaled_group, _ = AggregationConstraint('sum').rescale(
                values[rows], observations[rows], period_freq,
                grid_freq={(entity,): 'M' for entity in entities},
            )
            pieces.append(rescaled_group)
        rescaled = pd.concat(pieces)

        # Valeurs d'or : FR 120 / 12 = 10 par mois ; DE 1, 2, 3, 4 par mois selon le trimestre
        expected_de = [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3
        assert rescaled.loc['FR'].tolist() == [10.0] * 12
        assert rescaled.loc['DE'].tolist() == expected_de

    def test_grid_frequency_read_per_entity(self):
        """FR complete on months, DE truncated on quarters: only FR is rescaled."""
        months = _monthly_grid()
        quarters = pd.date_range('2021-03-31', periods=2, freq='QE')
        france = pd.Series(10.0, index=pd.MultiIndex.from_product(
            [['FR'], months], names=['country', 'date']))
        germany = pd.Series(30.0, index=pd.MultiIndex.from_product(
            [['DE'], quarters], names=['country', 'date']))
        values = pd.concat([france, germany])
        observations = pd.Series(np.nan, index=values.index)
        observations.loc[('FR', months[-1])] = 120.0
        observations.loc[('DE', quarters[-1])] = 120.0

        with pytest.warns(UserWarning, match=r'not enforced.*1 observed period\(s\)'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M', ('DE',): 'Q'}
            )

        assert mask.loc['FR'].all() and not mask.loc['DE'].any()
        assert rescaled.loc['DE'].tolist() == [30.0, 30.0]

    def test_three_level_index(self):
        """A (region, entity, date) index keeps one block per (region, entity)."""
        values = to_three_level_index(_to_panel(_raw_predictions()).to_frame('a1'))['a1']
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = -60.0
        observations = to_three_level_index(observations.to_frame('a1'))['a1']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y',
                grid_freq={('Zone euro', 'FR'): 'M', ('Zone euro', 'DE'): 'M'},
            )

        sums = rescaled.groupby(level=['region', 'country']).sum()
        assert sums.to_dict() == pytest.approx(
            {('Zone euro', 'FR'): 120.0, ('Zone euro', 'DE'): -60.0}, abs=_EXACT
        )

    @pytest.mark.parametrize(
        'perturb', [lambda s: shuffle_rows(s, seed=3), reverse_entities],
        ids=['shuffle_rows', 'reverse_entities'],
    )
    def test_row_order_does_not_change_any_cell(self, perturb):
        """Shuffled rows or reversed entities: same value for every cell."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = 240.0
        reference, _ = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        perturbed, _ = AggregationConstraint('sum').rescale(
            perturb(values), observations, 'Y', grid_freq='M'
        )

        pd.testing.assert_series_equal(perturbed.reindex(reference.index), reference)
