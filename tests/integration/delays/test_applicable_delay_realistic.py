"""Realistic scenarios for ``calculate_applicable_delay``.

Chains ``compare_and_detect_delays`` and ``calculate_applicable_delay`` on the
realistic datasets of notebook 3 (``heterogeneous_coverage_panel`` and
``irregular_index_timeseries``: per-entity coverage, a column whose frequency
depends on the entity, genuinely irregular indexes), as a user preparing the
publication delays of a forecasting model would.

The panel is downloaded for the first time on 2024-10-15 (``existing_data=None``):
the detected observation of each (entity, column) couple is its last non-null
value, dated on the first day of its period. The expected delays are derived by
hand from that date, with the calendar only, never from the output of the code:

* the delay is counted from the start of the observation period, which is the
  observation date;
* converted to a target frequency, the delay is counted again from the start or
  from the (exclusive) end of the target period containing the observation date,
  which is, for a source period starting on that date, the target period starting
  on it when the target frequency is higher (the first sub-period), and the
  target period that contains it when it is equal or lower.

Notably the same column (``depenses_publiques_pib``) is annual for two entities
and quarterly for the third, and ``climat_affaires`` is absent for one entity: a
single target frequency per indicator therefore converts rows of different
source frequencies.
"""
# Modules de base
import pandas as pd
import pytest

# Fonctions à tester
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (ANO-UTILS-033) : le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")

TS = pd.Timestamp
DOWNLOAD = TS('2024-10-15')


# =============================================================================
# Valeurs d'or dérivées du calendrier
# =============================================================================
def _last_observations(data: pd.DataFrame) -> dict:
    """Return the last non-null date of every (entity, column) couple, or of every column for a series.

    Args:
        data: Panel (two-level index: entity, date) or time series (date index).

    Returns:
        ``{(entity, column): date}`` for a panel, ``{column: date}`` for a series; couples without
        any observation are left out.
    """
    last = {}
    if isinstance(data.index, pd.MultiIndex):
        for entity in data.index.get_level_values(0).unique():
            for column in data.columns:
                date = data.loc[entity, column].last_valid_index()
                if date is not None:
                    last[(entity, column)] = date
    else:
        for column in data.columns:
            date = data[column].last_valid_index()
            if date is not None:
                last[column] = date
    return last


def _days_from_start_of_month(date: pd.Timestamp) -> int:
    """Return the days from the start of the month following ``date`` to the download date (month end, exclusive)."""
    return (DOWNLOAD - (date + pd.DateOffset(months=1))).days


def _days_from_start_of_quarter(date: pd.Timestamp) -> int:
    """Return the days from the start of the quarter containing ``date`` to the download date."""
    return (DOWNLOAD - date.to_period('Q').start_time).days


def _days_from_end_of_year(date: pd.Timestamp) -> int:
    """Return the days from the (exclusive) end of the year containing ``date`` to the download date (negative if later)."""
    return (DOWNLOAD - TS(year=date.year + 1, month=1, day=1)).days


@pytest.fixture
def panel_delays(heterogeneous_coverage_panel) -> pd.DataFrame:
    """First download of the heterogeneous panel on 2024-10-15, delays counted from the period start."""
    return compare_and_detect_delays(heterogeneous_coverage_panel, None, DOWNLOAD, reference_point='start')


@pytest.fixture
def series_delays(irregular_index_timeseries) -> pd.DataFrame:
    """First download of the irregular time series on 2024-10-15, delays counted from the period start."""
    return compare_and_detect_delays(irregular_index_timeseries, None, DOWNLOAD, reference_point='start')


# =============================================================================
# Panel hétérogène
# =============================================================================
class TestHeterogeneousPanel:
    """Per-entity and per-indicator delays of the notebook 3 panel."""

    def test_monthly_target_counted_from_the_end(self, heterogeneous_coverage_panel, panel_delays):
        """Monthly target, from the month end: every couple gets the days since the end of its first month.

        Annual and quarterly observations, dated on the first day of their period, fall in the first month.
        """
        result = calculate_applicable_delay(panel_delays, 'end', 'monthly', aggregate_by_panel=True)

        expected = {key: _days_from_start_of_month(date)
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_monthly_target_counted_from_the_start_is_the_detected_delay(self, heterogeneous_coverage_panel, panel_delays):
        """Monthly target from the start: for observations dated on a period start, the detected delay is unchanged."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M', aggregate_by_panel=True)

        expected = {key: (DOWNLOAD - date).days for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_quarterly_target_counted_from_the_start(self, heterogeneous_coverage_panel, panel_delays):
        """Quarterly target from the start: the quarter containing the observation (monthly sources) or starting on it."""
        result = calculate_applicable_delay(panel_delays, 'start', 'quarterly', aggregate_by_panel=True)

        expected = {key: _days_from_start_of_quarter(date)
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_annual_target_counted_from_the_end_can_be_negative(self, heterogeneous_coverage_panel, panel_delays):
        """Annual target from the year end: the 2024 observations are published before their year ends (negative delays)."""
        result = calculate_applicable_delay(panel_delays, 'end', 'annual', aggregate_by_panel=True)

        expected = {key: _days_from_end_of_year(date)
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_annual_target_counted_from_the_end_is_negative_for_the_current_year(self, panel_delays):
        """Observations of 2024 (Jan 1st 2025 - 15 Oct 2024 = -78 d at most) give negative delays, those of 2023 positive."""
        result = calculate_applicable_delay(panel_delays, 'end', 'annual', aggregate_by_panel=True)

        assert (result['delay'] < 0).any() and (result['delay'] > 0).any()

    def test_one_row_per_couple(self, heterogeneous_coverage_panel, panel_delays):
        """20 couples: 3 entities x 7 columns, ``climat_affaires`` being absent for Italy."""
        result = calculate_applicable_delay(panel_delays, 'end', 'M', aggregate_by_panel=True)

        assert (len(result), list(result.index.names)) == (20, ['country', 'column'])

    def test_median_over_the_entities_of_each_indicator(self, heterogeneous_coverage_panel, panel_delays):
        """Default aggregation: median over the countries of each column (two of them for ``climat_affaires``)."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M')

        by_column = pd.Series({key: (DOWNLOAD - date).days for key, date in
                               _last_observations(heterogeneous_coverage_panel).items()})
        expected = by_column.groupby(level=1).median()
        assert result['delay'].to_dict() == expected.to_dict()

    def test_number_of_observations_per_indicator(self, heterogeneous_coverage_panel, panel_delays):
        """Each column is counted once per entity that publishes it: 3, except ``climat_affaires`` (2)."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M')

        assert result['n_observations'].to_dict() == {
            'balance_commerciale_annuelle': 3, 'climat_affaires': 2, 'depenses_publiques_pib': 3,
            'inflation_ipc': 3, 'pib_trimestriel': 3, 'production_industrielle': 3, 'taux_chomage': 3,
        }

    def test_target_frequency_per_indicator(self, heterogeneous_coverage_panel, panel_delays):
        """One target frequency per indicator: monthly, quarterly (GDP and public spending) and annual (trade balance)."""
        target = {'production_industrielle': 'monthly', 'inflation_ipc': 'monthly', 'taux_chomage': 'monthly',
                  'climat_affaires': 'monthly', 'pib_trimestriel': 'quarterly', 'depenses_publiques_pib': 'quarterly',
                  'balance_commerciale_annuelle': 'annual'}
        last = _last_observations(heterogeneous_coverage_panel)

        result = calculate_applicable_delay(panel_delays, 'start', target, aggregate_by_panel=True)

        # Valeurs d'or : mensuel = début de la période de l'observation (la date elle-même) ; trimestriel = début
        # du trimestre contenant la date ; annuel = 1er janvier de son année
        start_of_period = {'monthly': lambda d: d, 'quarterly': lambda d: d.to_period('Q').start_time,
                           'annual': lambda d: TS(year=d.year, month=1, day=1)}
        expected = {key: (DOWNLOAD - start_of_period[target[key[1]]](date)).days for key, date in last.items()}
        assert result['delay'].to_dict() == expected

    def test_target_frequency_column_reports_each_indicator_frequency(self, panel_delays):
        """The ``frequency`` column holds the requested frequency of each indicator."""
        target = {column: 'quarterly' if column == 'pib_trimestriel' else 'monthly'
                  for column in panel_delays.index.get_level_values('column').unique()}

        result = calculate_applicable_delay(panel_delays, 'start', target)

        assert result['frequency'].to_dict() == target

    @pytest.mark.parametrize(
        ('unit', 'code', 'from_days'),
        [('hour', 'h', lambda days: days * 24), ('second', 's', lambda days: days * 86_400),
         ('week', 'W', lambda days: -(-days // 7))],
        ids=['hour', 'second', 'week'],
    )
    def test_target_unit(self, heterogeneous_coverage_panel, panel_delays, unit, code, from_days):
        """Whole days converted to a finer unit (exact) or a coarser one (rounded up to the next whole week)."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M', aggregate_by_panel=True, unit=unit)

        # Valeurs d'or : 24 h ou 86 400 s par jour ; une semaine = 7 jours, arrondie au supérieur
        expected = {key: from_days((DOWNLOAD - date).days)
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_target_unit_label_is_the_duration_code(self, panel_delays):
        """The unit column holds the duration code of the requested unit."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M', unit='hour')

        assert result['unit'].unique().tolist() == ['h']

    def test_rows_in_any_order(self, panel_delays):
        """Shuffled rows of the delays frame give the same result."""
        shuffled = panel_delays.sample(frac=1.0, random_state=11)

        pd.testing.assert_frame_equal(
            calculate_applicable_delay(shuffled, 'end', 'Q', aggregate_by_panel=True),
            calculate_applicable_delay(panel_delays, 'end', 'Q', aggregate_by_panel=True),
        )

    def test_indicator_selection(self, panel_delays):
        """Selecting two indicators leaves only their rows."""
        result = calculate_applicable_delay(panel_delays, 'end', 'M', indicators=['inflation_ipc', 'taux_chomage'])

        assert list(result.index) == ['inflation_ipc', 'taux_chomage']


# =============================================================================
# Série à index irrégulier
# =============================================================================
class TestIrregularTimeSeries:
    """Delays of the notebook 3 time series, whose annual anchors precede the monthly grid."""

    def test_monthly_target_counted_from_the_end(self, irregular_index_timeseries, series_delays):
        """One row per column, the isolated annual anchors not disturbing the monthly or quarterly columns."""
        result = calculate_applicable_delay(series_delays, 'end', 'M')

        expected = {column: _days_from_start_of_month(date)
                    for column, date in _last_observations(irregular_index_timeseries).items()}
        assert result['delay'].to_dict() == expected

    def test_quarterly_target_counted_from_the_start(self, irregular_index_timeseries, series_delays):
        """Quarterly target from the start of the quarter containing the last observation of each column."""
        result = calculate_applicable_delay(series_delays, 'start', 'Q')

        expected = {column: _days_from_start_of_quarter(date)
                    for column, date in _last_observations(irregular_index_timeseries).items()}
        assert result['delay'].to_dict() == expected

    def test_every_column_is_counted_once(self, series_delays):
        """A first download gives a single observation per column."""
        result = calculate_applicable_delay(series_delays, 'end', 'M')

        assert set(result['n_observations']) == {1}
