"""Rescaling of sub-periods onto the observed total.

Scenarios: sum invariant, hand-computed golden values per frequency pair and position,
realistic datasets and the mask of anchor cells.
"""

# Modules de base
import typing
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.aggregation_constraint import AggregationConstraint

# Perturbations partagées
from tests.support.perturbations import shuffle_rows

# Outils de test partagés du paquet
from tests.unit.frequency.aggregation_constraint.helpers import (
    _EXACT,
    _GRID_ALIASES,
    _RAW_2021,
    _annual_observations,
    _monthly_grid,
    _raw_predictions,
    _record_warnings,
)


# Couples (fréquence des totaux, base de la grille) du recalage
_FREQUENCY_PAIRS = {
    'Y->M': ('Y', 'M'),
    'Y->Q': ('Y', 'Q'),
    'Q->M': ('Q', 'M'),
}


# Fabrique des observations : une ancre par période, au début ou à la fin
def _anchor_observations(
    grid: pd.DatetimeIndex,
    period_freq: str,
    totals: typing.Sequence[float],
    position: str,
) -> pd.Series:
    """Place one observed total per period on its anchor row of the grid.

    Args:
        grid: Sorted grid covering whole periods.
        period_freq: Frequency of the periods (``'Y'`` or ``'Q'``).
        totals: One observed total per period, in chronological order.
        position: ``'start'`` anchors on the first row of each period (``YS``
            convention), ``'end'`` on the last one (``YE`` convention).

    Returns:
        ``Series`` on ``grid``, non-null on the anchor rows only.
    """
    observations = pd.Series(np.nan, index=grid)
    periods = grid.to_period(period_freq)
    for total, period in zip(totals, periods.unique()):
        rows = np.flatnonzero(periods == period)
        anchor = rows[0] if position == 'start' else rows[-1]
        observations.iloc[anchor] = total
    return observations


# Écart maximal entre la somme recalée de chaque période et son total observé
def _max_period_gap(
    rescaled: pd.Series, observations: pd.Series, period_freq: str
) -> float:
    """Return the largest |sum of rescaled sub-periods - observed total|.

    Args:
        rescaled: Rescaled values on a ``DatetimeIndex``.
        observations: Observed totals on their anchor rows.
        period_freq: Frequency of the periods.

    Returns:
        The worst gap over the periods holding an observation.
    """
    sums = rescaled.groupby(rescaled.index.to_period(period_freq)).sum()
    observed = observations.dropna()
    totals = observed.groupby(observed.index.to_period(period_freq)).sum()
    return float((sums.reindex(totals.index) - totals).abs().max())


# =============================================================================
# Recalage : invariant d'additivité par propriété
# =============================================================================
class TestRescaleSumInvariant:
    """Rescaled sub-periods sum to the observed total."""

    @pytest.mark.filterwarnings('ignore:Aggregation constraint flipped')
    @pytest.mark.parametrize('seed', range(5), ids=lambda seed: f'seed{seed}')
    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'pair', list(_FREQUENCY_PAIRS.values()), ids=list(_FREQUENCY_PAIRS)
    )
    def test_property_every_complete_period_sums_to_its_total(self, pair, position, seed):
        """Random predictions and totals of both signs: every period is exact to 1e-9.

        Three years of whole periods; per period, predictions of one random
        sign and magnitude 1-100, total drawn from N(0, 500) — so some periods
        are rescaled with a sign flip, which must not affect exactness.
        """
        period_freq, grid_base = pair
        rng = np.random.default_rng(seed)

        # Construction d'une grille couvrant exactement 2020-2022
        grid = pd.date_range(
            '2020-01-01', '2022-12-31', freq=_GRID_ALIASES[(grid_base, position)]
        )
        periods = grid.to_period(period_freq)
        n_periods = periods.nunique()

        # Prédictions : un signe par période, magnitudes strictement non nulles
        signs = rng.choice([-1.0, 1.0], size=n_periods)
        sign_by_row = signs[pd.factorize(periods)[0]]
        values = pd.Series(sign_by_row * rng.uniform(1.0, 100.0, len(grid)), index=grid)
        observations = _anchor_observations(
            grid, period_freq, rng.normal(0.0, 500.0, n_periods), position
        )

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert _max_period_gap(rescaled, observations, period_freq) <= _EXACT

    @pytest.mark.filterwarnings('ignore:Aggregation constraint flipped')
    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'pair', list(_FREQUENCY_PAIRS.values()), ids=list(_FREQUENCY_PAIRS)
    )
    def test_property_every_cell_is_moved(self, pair, position):
        """With whole periods and non-zero aggregates, the mask covers the whole grid."""
        period_freq, grid_base = pair
        rng = np.random.default_rng(0)
        grid = pd.date_range(
            '2020-01-01', '2022-12-31', freq=_GRID_ALIASES[(grid_base, position)]
        )
        n_periods = grid.to_period(period_freq).nunique()
        values = pd.Series(rng.uniform(1.0, 100.0, len(grid)), index=grid)
        observations = _anchor_observations(
            grid, period_freq, rng.normal(0.0, 500.0, n_periods), position
        )

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert mask.all()


class TestRescaleGoldenValues:
    """Hand-computed rescaling, for each frequency pair and position."""

    def test_reference_example_a1_120_in_2021(self, reference_timeseries):
        """Reference example: a1 = 120 in 2021, stage M, raw predictions summing to 112.5."""
        # Grille et observations tirées du jeu de référence
        grid = reference_timeseries.loc['2021'].index
        observations = reference_timeseries.loc['2021', 'a1']
        values = pd.Series(_RAW_2021, index=grid, name='a1')

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Valeur d'or : chaque mois multiplié par 120 / 112.5 = 1.0666…
        expected = [value * 120.0 / 112.5 for value in _RAW_2021]
        np.testing.assert_allclose(rescaled.to_numpy(), expected, rtol=0, atol=1e-12)
        # La ligne d'ancre porte une sous-période (11.2), jamais le total
        assert rescaled.loc['2021-12-31'] == pytest.approx(11.2)
        assert mask.all()

    # Cas d'or : (fréquence des totaux, base de grille, prédictions, total, attendu)
    _GOLDEN = {
        # Valeur d'or : 10+20+30+40 = 100, ratio 120 / 100 = 1.2
        'Y->Q': ('Y', 'Q', [10.0, 20.0, 30.0, 40.0], 120.0, [12.0, 24.0, 36.0, 48.0]),
        # Valeur d'or : 1+2+3 = 6, ratio 12 / 6 = 2
        'Q->M': ('Q', 'M', [1.0, 2.0, 3.0], 12.0, [2.0, 4.0, 6.0]),
        # Valeur d'or : douze mois à 5, somme 60, ratio 120 / 60 = 2
        'Y->M': ('Y', 'M', [5.0] * 12, 120.0, [10.0] * 12),
    }

    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'period_freq, grid_base, raw, total, expected',
        list(_GOLDEN.values()), ids=list(_GOLDEN),
    )
    def test_golden_rescaling(self, period_freq, grid_base, raw, total, expected, position):
        """Sub-periods are multiplied by total / aggregate, at either anchor position."""
        grid = pd.date_range(
            '2021-01-01', periods=len(raw), freq=_GRID_ALIASES[(grid_base, position)]
        )
        values = pd.Series(raw, index=grid)
        observations = _anchor_observations(grid, period_freq, [total], position)

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert rescaled.tolist() == expected

    def test_several_observations_of_one_period_are_summed(self):
        """Two observations falling in one period make a single target total.

        Pinned behaviour: the period frequency is a per-call setting. Quarterly
        observations rescaled with ``period_freq='Y'`` are summed into one
        annual target — the year is exact, the quarters are not. Hence the
        caller groups entities by source frequency (``HighFrequencyImputer``
        calls ``rescale`` with each step's own ``source_frequency``).
        """
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(np.nan, index=grid)
        # Valeur d'or : 4 trimestres de 30 → total annuel 120, 10 par mois
        observations.iloc[[2, 5, 8, 11]] = [30.0, 30.0, 30.0, 30.0]

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        assert rescaled.tolist() == [10.0] * 12


# =============================================================================
# Recalage : jeux réalistes
# =============================================================================
class TestRescaleRealisticDatasets:
    """Properties on the realistic datasets of notebook 3."""

    @staticmethod
    def _balance_case(irregular_index_timeseries: pd.DataFrame, sign: float):
        """Return random predictions of one sign and the negative annual trade balance."""
        observations = irregular_index_timeseries['balance_commerciale_annuelle']
        rng = np.random.default_rng(11)
        values = pd.Series(
            sign * rng.uniform(0.5, 5.0, len(observations)), index=observations.index
        )
        return values, observations

    def test_negative_annual_trade_balance_is_matched_exactly(self, irregular_index_timeseries):
        """Every complete year 2018-2023 sums to its NEGATIVE observed balance.

        Predictions are positive: each year is rescaled with a sign flip.
        """
        values, observations = self._balance_case(irregular_index_timeseries, +1.0)
        assert (observations.dropna() < 0).all()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='MS'
            )

        # Années complètes et observées de la grille mensuelle
        complete = observations.loc['2018':'2023'].dropna()
        sums = rescaled.groupby(rescaled.index.year).sum().loc[2018:2023]
        assert np.abs(sums.to_numpy() - complete.to_numpy()).max() <= _EXACT

    def test_isolated_anchors_and_unpublished_year_stay_raw(self, irregular_index_timeseries):
        """2015-2017 hold one row each, 2024 has no total: none of them moves."""
        values, observations = self._balance_case(irregular_index_timeseries, +1.0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='MS'
            )

        untouched = mask.index.year.isin([2015, 2016, 2017, 2024])
        assert not mask[untouched].any()
        pd.testing.assert_series_equal(rescaled[untouched], values[untouched])

    @pytest.mark.parametrize('sign, expected_warnings', [(+1.0, 1), (-1.0, 0)],
                             ids=['opposite-sign', 'same-sign'])
    def test_trade_balance_warns_once_for_all_flipped_years(
        self, irregular_index_timeseries, sign, expected_warnings
    ):
        """Six flipped years produce ONE warning; same-sign predictions none.

        The years the grid holds in part — the isolated anchors before the
        monthly history, the last one after it — are named by their own
        single warning, whatever the sign.
        """
        values, observations = self._balance_case(irregular_index_timeseries, sign)

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, observations, 'Y', grid_freq='MS',
        )

        flipped = [m for m in messages if 'flipped the predicted profile' in m]
        assert len(flipped) == expected_warnings
        if expected_warnings:
            assert '6 period(s)' in flipped[0]
        truncated = [m for m in messages if m.startswith('Aggregation constraint not enforced')]
        assert len(truncated) == 1 and len(messages) == len(flipped) + 1

    def test_per_entity_source_frequency_on_the_heterogeneous_panel(
        self, heterogeneous_coverage_panel
    ):
        """depenses_publiques_pib: annual (France, Italie) and quarterly (Allemagne) totals.

        One call per source-frequency group, as the imputer does; every
        observed period (6 French years, 5 Italian years, 23 German quarters,
        all whole on the monthly grid) sums to its total within 1e-9.
        """
        observations = heterogeneous_coverage_panel['depenses_publiques_pib']
        rng = np.random.default_rng(5)
        values = pd.Series(rng.uniform(1.0, 20.0, len(observations)), index=observations.index)
        countries = observations.index.get_level_values('country')

        gaps, n_periods = [], 0
        for period_freq, entities in {'Y': ['France', 'Italie'], 'Q': ['Allemagne']}.items():
            rows = countries.isin(entities)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values[rows], observations[rows], period_freq, grid_freq='MS'
            )
            for entity in entities:
                observed = observations.loc[entity].dropna()
                gaps.append(_max_period_gap(rescaled.loc[entity], observed, period_freq))
                n_periods += len(observed)

        assert n_periods == 6 + 5 + 23
        assert max(gaps) <= _EXACT


# =============================================================================
# Masque des ancres
# =============================================================================
class TestAnchorCellsMask:
    """Anchor rows re-expressed at the stage frequency.

    A DIAGNOSTIC mask: it locates the observed totals overwritten by a
    sub-period value, it qualifies no provenance.
    """

    def test_exactly_the_anchor_cells(self):
        """True on the anchor rows, and only there."""
        grid = pd.date_range('2021-01-01', periods=24, freq='MS')
        observations = pd.Series(np.nan, index=grid)
        anchors = pd.DatetimeIndex(['2021-01-01', '2022-01-01'])
        observations.loc[anchors] = [100.0, -3.0]

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, grid)

        assert mask[mask].index.equals(anchors)

    @pytest.mark.parametrize('setting', ['sum', None, {'a1': None}],
                             ids=['sum', 'none', 'dict'])
    def test_independent_of_the_constraint(self, reference_timeseries, setting):
        """Same mask under every setting: the three annual anchors of a1."""
        grid = reference_timeseries.index
        observations = reference_timeseries['a1']

        mask = AggregationConstraint(setting).anchor_cells_mask(observations, grid)

        assert mask[mask].index.equals(observations.dropna().index)

    def test_independent_of_the_rescaling_outcome(self):
        """A period left raw keeps its anchor located."""
        values = _raw_predictions()
        values.iloc[3] = np.nan
        observations = _annual_observations()

        constraint = AggregationConstraint('sum')
        _, rescaled_mask = constraint.rescale(values, observations, 'Y')
        anchor_mask = constraint.anchor_cells_mask(observations, values.index)

        assert not rescaled_mask.any()
        assert anchor_mask.tolist() == [False] * 11 + [True]

    def test_dataframe_input_gives_one_column_per_variable(self, reference_timeseries):
        """A frame of observations yields a boolean frame, each column its own anchors."""
        observations = reference_timeseries[['a1', 'q1', 'm1']]

        mask = AggregationConstraint('sum').anchor_cells_mask(
            observations, reference_timeseries.index
        )

        assert isinstance(mask, pd.DataFrame)
        pd.testing.assert_frame_equal(mask, observations.notna())

    def test_rows_off_the_grid_are_ignored_and_new_rows_are_false(self):
        """Anchors outside the grid vanish; grid rows without observation are False."""
        observations = pd.Series([120.0, 132.0], index=pd.to_datetime(['2020-12-31', '2021-12-31']))
        grid = _monthly_grid(2021)

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, grid)

        assert mask.tolist() == [False] * 11 + [True]

    def test_no_observation_gives_an_all_false_mask(self):
        """A column without anchor on the grid yields an all-False mask."""
        grid = _monthly_grid()
        observations = pd.Series(np.nan, index=grid, name='a1')

        assert not AggregationConstraint('sum').anchor_cells_mask(observations, grid).any()

    def test_panel_heterogeneous_anchors(self, heterogeneous_coverage_panel):
        """Panel: a MultiIndex Series, True on each entity's own anchors only.

        depenses_publiques_pib: 6 French and 5 Italian years, 23 German
        quarters; climat_affaires never observed by Italie.
        """
        observations = heterogeneous_coverage_panel[['depenses_publiques_pib', 'climat_affaires']]
        index = heterogeneous_coverage_panel.index

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, index)

        counts = mask['depenses_publiques_pib'].groupby(level='country').sum().to_dict()
        assert counts == {'Allemagne': 23, 'France': 6, 'Italie': 5}
        assert not mask.loc['Italie', 'climat_affaires'].any()

    def test_shuffled_grid_is_aligned_by_label(self, reference_timeseries):
        """A shuffled grid gets the same mask, row for row."""
        observations = reference_timeseries['a1']
        shuffled = shuffle_rows(reference_timeseries, seed=1).index

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, shuffled)

        pd.testing.assert_series_equal(mask, observations.notna().reindex(shuffled))

    def test_duplicated_observation_dates_raise(self):
        """Duplicated observation dates raise, with the message of the rescaling."""
        grid = _monthly_grid()
        observations = pd.Series(120.0, index=grid[[-1, -1]])

        with pytest.raises(ValueError, match='Duplicate dates in observations'):
            AggregationConstraint('sum').anchor_cells_mask(observations, grid)
