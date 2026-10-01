"""``FrequencyConverter`` on the realistic datasets of notebook 3.

Every public method (``convert_frequency``, ``convert``,
``aggregate_to_lower_frequency``, ``interpolate_to_higher_frequency``,
``count_subperiods_per_period``, ``get_conversion_factor``) runs at least once
on ``irregular_index_timeseries`` (mixed frequencies, annual anchors before the
monthly grid: a genuinely irregular index) and on
``heterogeneous_coverage_panel`` (coverage per entity, ``depenses_publiques_pib``
annual for France / Italie and quarterly for Allemagne, ``climat_affaires``
never observed for Italie).

No golden value is written by hand on these datasets: each expectation is
recomputed independently with plain pandas (``groupby`` by year, ``reindex`` +
``interpolate`` on the observed grid) and the tests check properties —
totals and means preserved, observations kept, no gap inside the observed
range, sub-period counts consistent with the detected frequencies. The
corrections of the DataFrame path (ANO-UTILS-054, 055, 065) are checked on
both datasets.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter
from tsforecast.utils.frequency.utils import detect_dataset_frequency, detect_frequency

# Colonnes mensuelles des deux jeux (grille MS)
MONTHLY_COLUMNS = ['production_industrielle', 'inflation_ipc', 'taux_chomage']

# Nombre d'observations attendu par année pleine, selon la fréquence détectée
# (valeurs d'or de la détection du prompt U7, voir detector/test_realistic_datasets.py)
EXPECTED_YEARLY_FACTORS = {'MS': 12.0, 'QS-JAN': 4.0, 'YS-JAN': 1.0}


@pytest.fixture
def converter() -> FrequencyConverter:
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


def _yearly_reduction(observed: pd.DataFrame, how: str, required: int = 0) -> pd.DataFrame:
    """Reduce each column by calendar year, independently of the converter.

    Args:
        observed: Date-indexed values (NaN rows allowed).
        how: ``'mean'`` or ``'sum'``.
        required: Minimum number of observations per year; years below it are
            NaN (``0``: no requirement).

    Returns:
        One row per calendar year from the first to the last row of
        ``observed``, indexed by the first day of the year.
    """
    by_year = observed.groupby(observed.index.year)
    reduced = getattr(by_year, how)().where(by_year.count() >= required)
    years = range(observed.index.min().year, observed.index.max().year + 1)
    reduced = reduced.reindex(years)
    reduced.index = pd.to_datetime([f'{year}-01-01' for year in years])
    return reduced


def _interior_yearly_counts(series: pd.Series) -> set:
    """Distinct numbers of observations per year, first and last observed years excluded."""
    observed = series.dropna()
    counts = observed.groupby(observed.index.year).size()
    interior = counts.loc[observed.index.year.min() + 1:observed.index.year.max() - 1]
    return set(interior.astype(float))


def _quarter_starts_only(index: pd.Index) -> bool:
    """True iff every date of the index (last level) is the first day of a quarter."""
    dates = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
    return bool(((dates.day == 1) & (dates.month % 3 == 1)).all())


# =============================================================================
# irregular_index_timeseries
# =============================================================================

class TestIrregularIndexTimeseries:
    """Each public method on the genuinely irregular mixed-frequency time series."""

    def test_convert_frequency_yearly_means_of_complete_years(self, converter, irregular_index_timeseries):
        """Monthly columns → yearly means, incomplete years masked, per column."""
        monthly = irregular_index_timeseries[MONTHLY_COLUMNS]
        result = converter.convert_frequency(monthly, 'YS', method='mean', full_periods_only=True)
        expected = _yearly_reduction(monthly, 'mean', required=12)
        pd.testing.assert_frame_equal(result, expected, check_freq=False, check_names=False)

    def test_convert_delegates(self, converter, irregular_index_timeseries):
        """``convert`` gives the same yearly means as ``convert_frequency``."""
        monthly = irregular_index_timeseries[MONTHLY_COLUMNS]
        result = converter.convert(monthly, 'monthly', 'YS', method='mean', full_periods_only=True)
        expected = _yearly_reduction(monthly, 'mean', required=12)
        pd.testing.assert_frame_equal(result, expected, check_freq=False, check_names=False)

    @pytest.mark.parametrize('column', ['pib_trimestriel', 'balance_commerciale_annuelle'])
    def test_convert_frequency_upsampling_keeps_observations(self, converter, irregular_index_timeseries, column):
        """Quarterly and annual columns → months: every observation is kept at its date."""
        frame = irregular_index_timeseries[['pib_trimestriel', 'balance_commerciale_annuelle']]
        result = converter.convert_frequency(frame, 'MS', method='linear')
        observed = frame[column].dropna()
        pd.testing.assert_series_equal(result.loc[observed.index, column], observed, check_names=False)

    @pytest.mark.parametrize('column', ['pib_trimestriel', 'balance_commerciale_annuelle'])
    def test_convert_frequency_upsampling_leaves_no_inner_gap(self, converter, irregular_index_timeseries, column):
        """Between the first and last observations, every month is filled."""
        frame = irregular_index_timeseries[['pib_trimestriel', 'balance_commerciale_annuelle']]
        result = converter.convert_frequency(frame, 'MS', method='linear')
        observed = frame[column].dropna()
        assert result.loc[observed.index[0]:observed.index[-1], column].notna().all()

    @pytest.mark.parametrize(
        'column, source_freq, per_year',
        [
            pytest.param('production_industrielle', 'MS', 12, id='monthly'),
            pytest.param('inflation_ipc', 'MS', 12, id='monthly-last-missing'),
            pytest.param('pib_trimestriel', 'QS', 4, id='quarterly'),
            pytest.param('balance_commerciale_annuelle', 'YS', 1, id='annual'),
        ],
    )
    def test_aggregate_observations_to_yearly_totals(
        self, converter, irregular_index_timeseries, column, source_freq, per_year
    ):
        """The observed values of each column sum to their complete yearly totals."""
        observed = irregular_index_timeseries[column].dropna()
        result = converter.aggregate_to_lower_frequency(
            observed, 'YS', method='sum', full_periods_only=True, source_freq=source_freq
        )
        expected = _yearly_reduction(observed.to_frame(), 'sum', required=per_year)[column]
        pd.testing.assert_series_equal(result, expected, check_freq=False, check_names=False)

    def test_interpolate_quarterly_observations_to_months(self, converter, irregular_index_timeseries):
        """Quarterly GDP → months, inside only: the linear path of plain pandas on the observed grid."""
        observed = irregular_index_timeseries['pib_trimestriel'].dropna()
        result = converter.interpolate_to_higher_frequency(observed, 'MS', method='linear', limit_area='inside')

        months = pd.date_range(observed.index[0], observed.index[-1], freq='MS')
        expected = observed.reindex(months).interpolate(method='linear')
        pd.testing.assert_series_equal(
            result.loc[months[0]:months[-1]], expected, check_freq=False, check_names=False
        )

    def test_count_subperiods_bounds_the_monthly_observations(self, converter, irregular_index_timeseries):
        """No year holds more monthly observations than its calendar count of months."""
        inflation = irregular_index_timeseries['inflation_ipc'].dropna()
        observed_counts = inflation.groupby(inflation.index.year).size()
        years = pd.to_datetime([f'{year}-01-01' for year in observed_counts.index])

        expected_counts = converter.count_subperiods_per_period(years, 'YS', 'MS')

        assert (observed_counts.to_numpy() <= expected_counts).all()

    @pytest.mark.parametrize('column', MONTHLY_COLUMNS + ['pib_trimestriel', 'balance_commerciale_annuelle'])
    def test_conversion_factor_counts_interior_years(self, converter, irregular_index_timeseries, column):
        """Each interior year holds exactly ``get_conversion_factor(freq, 'YS')`` observations."""
        series = irregular_index_timeseries[column]
        frequency = detect_frequency(series, return_format='full')
        factor = converter.get_conversion_factor(frequency, 'YS')
        assert _interior_yearly_counts(series) == {factor} == {EXPECTED_YEARLY_FACTORS[frequency]}

    def test_annual_column_to_quarters(self, converter, irregular_index_timeseries):
        """The annual trade balance interpolates to quarter starts."""
        frame = irregular_index_timeseries[['balance_commerciale_annuelle']]
        result = converter.convert_frequency(frame, 'QS', method='linear')
        assert _quarter_starts_only(result.index)

    def test_quarterly_output_has_quarter_starts_only(self, converter, irregular_index_timeseries):
        """Monthly production and quarterly GDP → quarters: only quarter starts remain."""
        frame = irregular_index_timeseries[['production_industrielle', 'pib_trimestriel']]
        result = converter.convert_frequency(frame, 'QS', method='mean')
        assert _quarter_starts_only(result.index)

    def test_years_without_observation_do_not_sum_to_zero(self, converter, irregular_index_timeseries):
        """Industrial production starts in 2019: 2015-2018 have no yearly total."""
        frame = irregular_index_timeseries[['production_industrielle']]
        result = converter.convert_frequency(frame, 'YS', method='sum')
        assert result.loc['2015':'2018', 'production_industrielle'].isna().all()


# =============================================================================
# heterogeneous_coverage_panel
# =============================================================================

ENTITIES = ['Allemagne', 'France', 'Italie']


def _per_entity(panel: pd.DataFrame, reduce) -> pd.DataFrame:
    """Apply a per-entity reduction and rebuild the (country, date) panel index."""
    parts = {entity: reduce(panel.loc[entity]) for entity in ENTITIES}
    return pd.concat(parts, names=['country', 'date'])


class TestHeterogeneousCoveragePanel:
    """Each public method on the heterogeneous panel of notebook 3."""

    def test_convert_frequency_yearly_means_per_entity(self, converter, heterogeneous_coverage_panel):
        """Monthly columns → yearly means of complete years, each entity on its own coverage."""
        panel = heterogeneous_coverage_panel[['inflation_ipc', 'taux_chomage']]
        result = converter.convert_frequency(panel, 'YS', method='mean', full_periods_only=True)
        expected = _per_entity(panel, lambda frame: _yearly_reduction(frame, 'mean', required=12))
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_convert_delegates(self, converter, heterogeneous_coverage_panel):
        """``convert`` gives the same yearly means as ``convert_frequency``."""
        panel = heterogeneous_coverage_panel[['inflation_ipc']]
        result = converter.convert(panel, 'monthly', 'YS', method='mean', full_periods_only=True)
        expected = _per_entity(panel, lambda frame: _yearly_reduction(frame, 'mean', required=12))
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_entity_target_converts_that_entity_only(self, converter, heterogeneous_coverage_panel):
        """Quarterly spending of Allemagne → yearly means ; France and Italie untouched."""
        panel = heterogeneous_coverage_panel[['depenses_publiques_pib']]
        result = converter.convert_frequency(panel, {('Allemagne',): 'YS'}, method='mean')

        expected_germany = _yearly_reduction(panel.loc['Allemagne'], 'mean')
        pd.testing.assert_frame_equal(result.loc['Allemagne'], expected_germany, check_freq=False, check_names=False)
        pd.testing.assert_frame_equal(result.loc[['France', 'Italie']], panel.loc[['France', 'Italie']])

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_aggregate_gdp_to_yearly_totals(self, converter, heterogeneous_coverage_panel, entity):
        """Quarterly GDP of each entity sums to its complete yearly totals."""
        observed = heterogeneous_coverage_panel.loc[entity, 'pib_trimestriel'].dropna()
        result = converter.aggregate_to_lower_frequency(
            observed, 'YS', method='sum', full_periods_only=True, source_freq='QS'
        )
        expected = _yearly_reduction(observed.to_frame(), 'sum', required=4)['pib_trimestriel']
        pd.testing.assert_series_equal(result, expected, check_freq=False, check_names=False)

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_interpolate_spending_to_quarters(self, converter, heterogeneous_coverage_panel, entity):
        """Spending (annual or quarterly by entity) → quarters: the linear path between observations."""
        observed = heterogeneous_coverage_panel.loc[entity, 'depenses_publiques_pib'].dropna()
        result = converter.interpolate_to_higher_frequency(observed, 'QS', method='linear', limit_area='inside')

        quarters = pd.date_range(observed.index[0], observed.index[-1], freq='QS')
        expected = observed.reindex(quarters).interpolate(method='linear')
        pd.testing.assert_series_equal(
            result.loc[quarters[0]:quarters[-1]], expected, check_freq=False, check_names=False
        )

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_count_subperiods_bounds_the_quarterly_observations(self, converter, heterogeneous_coverage_panel, entity):
        """No year of any entity holds more GDP releases than its four quarters."""
        gdp = heterogeneous_coverage_panel.loc[entity, 'pib_trimestriel'].dropna()
        observed_counts = gdp.groupby(gdp.index.year).size()
        years = pd.to_datetime([f'{year}-01-01' for year in observed_counts.index])

        expected_counts = converter.count_subperiods_per_period(years, 'YS', 'QS')

        assert (observed_counts.to_numpy() <= expected_counts).all()

    def test_conversion_factor_counts_interior_years(self, converter, heterogeneous_coverage_panel):
        """For every (entity, column), interior years hold ``get_conversion_factor(freq, 'YS')`` observations."""
        frequencies: Dict[Tuple[str, str], str] = detect_dataset_frequency(
            heterogeneous_coverage_panel, return_format='full'
        )
        observed = {}
        expected = {}
        for (entity, column), frequency in frequencies.items():
            # Couple jamais observé (climat_affaires / Italie) : aucune fréquence
            if frequency is None:
                continue
            series = heterogeneous_coverage_panel.loc[entity, column]
            observed[(entity, column)] = _interior_yearly_counts(series)
            expected[(entity, column)] = {converter.get_conversion_factor(frequency, 'YS')}

        # 20 couples : 21 moins climat_affaires pour l'Italie
        assert len(observed) == 20 and observed == expected

    def test_never_observed_column_does_not_keep_the_monthly_grid(self, converter, heterogeneous_coverage_panel):
        """Inflation and business climate → quarters: Italie has quarter starts only."""
        panel = heterogeneous_coverage_panel[['inflation_ipc', 'climat_affaires']]
        result = converter.convert_frequency(panel, 'QS', method='mean')
        assert _quarter_starts_only(result.loc[['Italie']].index)

    def test_annual_spending_to_quarters(self, converter, heterogeneous_coverage_panel):
        """France's annual spending interpolates to quarter starts."""
        panel = heterogeneous_coverage_panel[['depenses_publiques_pib']]
        result = converter.convert_frequency(panel, {('France',): 'QS'}, method='linear')
        assert _quarter_starts_only(result.loc[['France']].index)
