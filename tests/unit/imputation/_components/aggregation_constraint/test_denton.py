"""Additive Denton benchmarking of ``AggregationConstraint`` (``aggregation_constraint='denton'``).

Scenarios: exact totals, absence of step at the period boundaries against pro rata, a constant gap
carried as a pure shift, the closed-form optimum of a small case, a variable crossing zero (no zero
nor flipped period), the guards shared with ``'sum'`` (truncated, partly predicted and unobserved
periods), entity isolation on a panel, idempotence, the dict form per column, the sklearn protocol,
and the solver of :func:`_denton_correction` alone.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.imputation._components.aggregation_constraint import (
    AggregationConstraint,
    _denton_correction,
)

# Outils de test partagés du paquet
from tests.unit.imputation._components.aggregation_constraint.helpers import (
    _EXACT,
    _record_warnings,
    _to_panel,
)


# =============================================================================
# Constructeurs
# =============================================================================
# Grille mensuelle de plusieurs années
def _months(years: int = 3, start: int = 2021) -> pd.DatetimeIndex:
    """Build the month-end dates of ``years`` consecutive years."""
    return pd.date_range(f'{start}-01-31', periods=12 * years, freq='ME')


# Observations annuelles, ancrées en décembre
def _annual(grid: pd.DatetimeIndex, totals) -> pd.Series:
    """Place one annual total on the December row of each year of the grid."""
    observations = pd.Series(np.nan, index=grid, name='a1')
    for position, total in zip(range(11, len(grid), 12), totals):
        observations.iloc[position] = total
    return observations


# Sommes annuelles d'une série
def _yearly_sums(values: pd.Series) -> list:
    """Sum a monthly series by calendar year."""
    dates = values.index.get_level_values(-1)
    return values.groupby(dates.year).sum().tolist()


# =============================================================================
# Totaux et forme
# =============================================================================
class TestTotalsAndShape:
    """Every complete period meets its total, without any step at its boundaries."""

    def test_every_complete_year_meets_its_total(self):
        """Three years of noisy predictions sum exactly to their three totals."""
        grid = _months(3)
        values = pd.Series(np.random.default_rng(0).normal(10.0, 1.0, len(grid)), index=grid)
        rescaled, mask = AggregationConstraint('denton').rescale(
            values, _annual(grid, [120.0, 150.0, 90.0]), 'Y'
        )
        assert np.allclose(_yearly_sums(rescaled), [120.0, 150.0, 90.0], atol=_EXACT)
        assert mask.all()

    def test_no_step_where_pro_rata_steps(self):
        """Flat predictions, totals doubling: pro rata steps by 10, Denton follows its slope."""
        grid = _months(2)
        values = pd.Series(10.0, index=grid)
        observations = _annual(grid, [120.0, 240.0])
        pro_rata, _ = AggregationConstraint('sum').rescale(values, observations, 'Y')
        smooth, _ = AggregationConstraint('denton').rescale(values, observations, 'Y')

        boundary = slice(11, 13)
        assert np.diff(pro_rata.iloc[boundary]).item() == pytest.approx(10.0)
        # Denton : le pas de la frontière est celui de la pente, comparable aux
        # pas voisins de chaque année, sans marche
        moves = np.abs(np.diff(smooth.to_numpy()))
        assert moves[11] < 2.0
        assert moves[11] == pytest.approx(moves[10], abs=0.2)
        assert moves[11] == pytest.approx(moves[12], abs=0.2)

    def test_constant_gap_is_a_pure_shift(self):
        """Every year short by the same 12: each month is raised by exactly 1."""
        grid = _months(3)
        values = pd.Series(np.arange(len(grid), dtype=float), index=grid)
        totals = [float(values.iloc[i:i + 12].sum()) + 12.0 for i in (0, 12, 24)]
        rescaled, _ = AggregationConstraint('denton').rescale(values, _annual(grid, totals), 'Y')
        assert np.allclose((rescaled - values).to_numpy(), 1.0, atol=_EXACT)

    def test_moves_of_the_predictions_are_kept_inside_a_period(self):
        """Within a year, the corrected series moves like the predictions, up to a smooth gap."""
        grid = _months(2)
        rng = np.random.default_rng(1)
        values = pd.Series(10.0 + rng.normal(0.0, 2.0, len(grid)), index=grid)
        rescaled, _ = AggregationConstraint('denton').rescale(
            values, _annual(grid, [150.0, 100.0]), 'Y'
        )
        correction = (rescaled - values).to_numpy()
        # La correction est lisse : ses variations sont bien plus faibles que celles des prédictions
        assert np.abs(np.diff(correction)).max() < np.abs(np.diff(values.to_numpy())).mean()

    def test_variable_crossing_zero_has_no_degenerate_period(self):
        """A year predicted around zero against a positive total: rescaled, no warning."""
        grid = _months(2)
        values = pd.Series(np.tile([1.0, -1.0], 12), index=grid)
        (rescaled, _mask), messages = _record_warnings(
            AggregationConstraint('denton').rescale, values, _annual(grid, [6.0, -6.0]), 'Y'
        )
        assert messages == []
        assert np.allclose(_yearly_sums(rescaled), [6.0, -6.0], atol=_EXACT)


# =============================================================================
# Gardes partagées avec 'sum'
# =============================================================================
class TestSharedGuards:
    """The periods constrained are those of ``'sum'``; the others receive a flat correction."""

    def test_unobserved_year_carries_the_last_gap_flat(self):
        """A year without total gets the correction of the last observed year, constant."""
        grid = _months(2)
        values = pd.Series(10.0, index=grid)
        observations = _annual(grid, [132.0])
        rescaled, _ = AggregationConstraint('denton').rescale(values, observations, 'Y')
        assert np.allclose(_yearly_sums(rescaled)[0], 132.0, atol=_EXACT)
        # Deuxième année : aucun total imposé, correction constante de 1 par mois
        assert np.allclose(rescaled.iloc[12:].to_numpy(), 11.0, atol=_EXACT)

    def test_truncated_year_is_not_imposed_and_is_reported(self):
        """Ten months of a year declared monthly: total not imposed, one aggregated warning."""
        grid = pd.date_range('2021-03-31', periods=22, freq='ME')
        values = pd.Series(10.0, index=grid)
        observations = pd.Series(np.nan, index=grid)
        observations.loc['2021-12-31'] = 150.0
        observations.loc['2022-12-31'] = 132.0
        (rescaled, _), messages = _record_warnings(
            AggregationConstraint('denton').rescale, values, observations, 'Y', grid_freq='M'
        )
        assert rescaled.loc['2022'].sum() == pytest.approx(132.0, abs=_EXACT)
        assert rescaled.loc['2021'].sum() != pytest.approx(150.0)
        assert len(messages) == 1 and 'not enforced' in messages[0]

    def test_partly_predicted_year_is_not_imposed(self):
        """A NaN month frees its year; the NaN cell stays NaN."""
        grid = _months(2)
        values = pd.Series(10.0, index=grid)
        values.iloc[3] = np.nan
        rescaled, _ = AggregationConstraint('denton').rescale(
            values, _annual(grid, [200.0, 132.0]), 'Y'
        )
        assert np.isnan(rescaled.iloc[3])
        assert rescaled.iloc[12:].sum() == pytest.approx(132.0, abs=_EXACT)
        assert rescaled.iloc[:12].sum() != pytest.approx(200.0)

    def test_no_observation_leaves_the_values_untouched(self):
        """Without any total, nothing moves and the mask is all-False."""
        grid = _months(1)
        values = pd.Series(10.0, index=grid)
        rescaled, mask = AggregationConstraint('denton').rescale(
            values, pd.Series(np.nan, index=grid), 'Y'
        )
        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_idempotent(self):
        """Benchmarking a benchmarked series changes nothing beyond rounding."""
        grid = _months(3)
        values = pd.Series(np.random.default_rng(2).normal(10.0, 1.0, len(grid)), index=grid)
        observations = _annual(grid, [130.0, 110.0, 140.0])
        constraint = AggregationConstraint('denton')
        once, _ = constraint.rescale(values, observations, 'Y')
        twice, _ = constraint.rescale(once, observations, 'Y')
        assert np.allclose(once.to_numpy(), twice.to_numpy(), atol=_EXACT)


# =============================================================================
# Panel, forme dictionnaire, protocole sklearn
# =============================================================================
class TestPanelAndForms:
    """Entities never share a correction; the setting resolves per column."""

    def test_entities_are_benchmarked_independently(self):
        """FR alone carries a gap: DE is untouched, FR meets its totals."""
        grid = _months(2)
        values = _to_panel(pd.Series(10.0, index=grid))
        observations = _to_panel(_annual(grid, [120.0, 120.0]))
        observations.loc[('FR', grid[11])] = 180.0
        rescaled, _ = AggregationConstraint('denton').rescale(values, observations, 'Y')
        assert np.allclose(rescaled.loc['DE'].to_numpy(), 10.0, atol=_EXACT)
        assert np.allclose(_yearly_sums(rescaled.loc['FR']), [180.0, 120.0], atol=_EXACT)

    def test_unsorted_rows_are_benchmarked_in_date_order(self):
        """Shuffled rows give the same values, date for date."""
        grid = _months(2)
        values = pd.Series(np.random.default_rng(3).normal(10.0, 1.0, len(grid)), index=grid)
        observations = _annual(grid, [130.0, 110.0])
        order = np.random.default_rng(4).permutation(len(grid))
        constraint = AggregationConstraint('denton')
        sorted_result, _ = constraint.rescale(values, observations, 'Y')
        shuffled_result, _ = constraint.rescale(values.iloc[order], observations, 'Y')
        pd.testing.assert_series_equal(
            shuffled_result.sort_index(), sorted_result, atol=_EXACT, check_freq=False
        )

    def test_dict_form_selects_the_constraint_per_column(self):
        """``{'a1': 'denton', '__default__': 'sum'}``: a1 smooth, a2 pro rata."""
        grid = _months(2)
        values = pd.Series(10.0, index=grid)
        observations = _annual(grid, [120.0, 240.0])
        constraint = AggregationConstraint({'a1': 'denton', '__default__': 'sum'})
        smooth, _ = constraint.rescale(values, observations, 'Y', column='a1')
        pro_rata, _ = constraint.rescale(values, observations, 'Y', column='a2')
        assert abs(smooth.iloc[12] - smooth.iloc[11]) < 2.0
        assert pro_rata.iloc[12] - pro_rata.iloc[11] == pytest.approx(10.0)

    def test_transform_benchmarks_every_column(self):
        """The sklearn protocol applies the setting of each column."""
        grid = _months(2)
        X = pd.DataFrame({'a1': 10.0, 'a2': 5.0}, index=grid)
        observations = pd.DataFrame({
            'a1': _annual(grid, [120.0, 240.0]), 'a2': _annual(grid, [60.0, 120.0]),
        })
        constraint = AggregationConstraint(
            'denton', period_frequencies='Y', observations=observations
        ).fit(X)
        result = constraint.transform(X)
        assert np.allclose(_yearly_sums(result['a1']), [120.0, 240.0], atol=_EXACT)
        assert np.allclose(_yearly_sums(result['a2']), [60.0, 120.0], atol=_EXACT)
        assert constraint.rescaled_mask_.all().all()


# =============================================================================
# Solveur
# =============================================================================
class TestDentonCorrection:
    """The correction is the constrained minimum of the squared first differences."""

    def test_closed_form_of_two_blocks(self):
        """Two blocks of two short by 2 and 6: correction 2/3, 4/3, 8/3, 10/3."""
        correction = _denton_correction(
            np.ones(4), [(np.array([0, 1]), 4.0), (np.array([2, 3]), 8.0)]
        )
        assert np.allclose(correction, [2 / 3, 4 / 3, 8 / 3, 10 / 3], atol=_EXACT)

    def test_single_value(self):
        """One value, one block: the correction is the whole gap."""
        assert _denton_correction(np.array([3.0]), [(np.array([0]), 5.0)]).tolist() == [2.0]

    def test_no_feasible_correction_is_smoother(self):
        """A perturbation keeping the totals never lowers the penalty."""
        preliminary = np.random.default_rng(5).normal(0.0, 1.0, 24)
        blocks = [(np.arange(0, 12), 5.0), (np.arange(12, 24), -3.0)]
        correction = _denton_correction(preliminary, blocks)
        penalty = np.sum(np.diff(correction) ** 2)
        rng = np.random.default_rng(6)
        for _ in range(20):
            noise = rng.normal(0.0, 0.1, 24)
            # Perturbation de somme nulle sur chaque bloc : totaux préservés
            for positions, _total in blocks:
                noise[positions] -= noise[positions].mean()
            assert np.sum(np.diff(correction + noise) ** 2) >= penalty - _EXACT
