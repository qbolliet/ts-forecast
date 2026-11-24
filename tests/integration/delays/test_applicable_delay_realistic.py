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
* converted to a target frequency, a period is published with its last
  sub-period, every sub-period at the same lag after its end: from the end, the
  delay is the lag between the (exclusive) end of the observation period, at its
  own frequency, and the download, at any target frequency; from the start, it
  adds the length of the target period containing the last day of the
  observation period (the last month of a quarter, the quarter of a month).

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
from tsforecast.utils.frequency import normalize_frequency

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index : le bruit est masqué pour garder une sortie lisible.
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


# Code de période pandas d'une fréquence détectée (ancrages calendaires par défaut dans ces jeux)
def _period_code(frequency: str) -> str:
    """Return the pandas period code ('M', 'Q', 'Y') of a detected frequency ('MS', 'QS-JAN', 'YS-JAN')."""
    return normalize_frequency(frequency)


def _days_after_end(date: pd.Timestamp, frequency: str) -> int:
    """Return the days from the (exclusive) end of the period of ``date`` at ``frequency`` to the download date."""
    end = (date.to_period(_period_code(frequency)) + 1).start_time
    return (DOWNLOAD - end).days


def _days_after_start(date: pd.Timestamp, frequency: str, target: str) -> int:
    """Return the days from the start of the ``target`` period containing the last day of the period of ``date``.

    Args:
        date: Observation date (start of its period).
        frequency: Detected frequency of the observation ('MS', 'QS-JAN', 'YS-JAN').
        target: Pandas period code of the target frequency ('M', 'Q', 'Y').

    Returns:
        The lag after the end of the observation period plus the length of that target period.
    """
    last_day = (date.to_period(_period_code(frequency)) + 1).start_time - pd.Timedelta(days=1)
    period = last_day.to_period(target)
    return _days_after_end(date, frequency) + ((period + 1).start_time - period.start_time).days


def _frequencies(delays: pd.DataFrame) -> dict:
    """Return the detected frequency of each couple (or column) of a first download, as given to the calculator."""
    return delays['frequency'].to_dict()


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
        """Monthly target, from the end: every couple gets the days since the end of its own period.

        A quarterly or annual observation is published with its last month, whose end is the end of the period.
        """
        result = calculate_applicable_delay(panel_delays, 'end', 'monthly', aggregate_by_panel=True)

        frequencies = _frequencies(panel_delays)
        expected = {key: _days_after_end(date, frequencies[key])
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_monthly_target_counted_from_the_start_of_the_last_month(self, heterogeneous_coverage_panel, panel_delays):
        """Monthly target from the start: the start of the last month of each observation period.

        For monthly couples, the detected delay is unchanged; quarterly and annual ones count from the start of
        March, June, ... or December.
        """
        result = calculate_applicable_delay(panel_delays, 'start', 'M', aggregate_by_panel=True)

        frequencies = _frequencies(panel_delays)
        expected = {key: _days_after_start(date, frequencies[key], 'M')
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_quarterly_target_counted_from_the_start(self, heterogeneous_coverage_panel, panel_delays):
        """Quarterly target from the start: the quarter containing the last day of each observation period."""
        result = calculate_applicable_delay(panel_delays, 'start', 'quarterly', aggregate_by_panel=True)

        frequencies = _frequencies(panel_delays)
        expected = {key: _days_after_start(date, frequencies[key], 'Q')
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    def test_annual_target_counted_from_the_start(self, heterogeneous_coverage_panel, panel_delays):
        """Annual target from the start: the year containing the last day of each observation period."""
        result = calculate_applicable_delay(panel_delays, 'start', 'annual', aggregate_by_panel=True)

        frequencies = _frequencies(panel_delays)
        expected = {key: _days_after_start(date, frequencies[key], 'Y')
                    for key, date in _last_observations(heterogeneous_coverage_panel).items()}
        assert result['delay'].to_dict() == expected

    @pytest.mark.parametrize('target', ['monthly', 'quarterly', 'annual'])
    def test_delay_from_the_end_does_not_depend_on_the_target_frequency(self, panel_delays, target):
        """From the end, the delay is the lag after the end of the observation period: the same at every target."""
        result = calculate_applicable_delay(panel_delays, 'end', target, aggregate_by_panel=True)
        own = calculate_applicable_delay(panel_delays, 'end', 'monthly', aggregate_by_panel=True)

        assert result['delay'].to_dict() == own['delay'].to_dict()

    def test_one_row_per_couple(self, heterogeneous_coverage_panel, panel_delays):
        """20 couples: 3 entities x 7 columns, ``climat_affaires`` being absent for Italy."""
        result = calculate_applicable_delay(panel_delays, 'end', 'M', aggregate_by_panel=True)

        assert (len(result), list(result.index.names)) == (20, ['country', 'column'])

    def test_median_over_the_entities_of_each_indicator(self, heterogeneous_coverage_panel, panel_delays):
        """Default aggregation: median over the countries of each column (two of them for ``climat_affaires``)."""
        result = calculate_applicable_delay(panel_delays, 'start', 'M')

        frequencies = _frequencies(panel_delays)
        by_column = pd.Series({key: _days_after_start(date, frequencies[key], 'M') for key, date in
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

        # Valeurs d'or : début de la période cible contenant le dernier jour de la période de l'observation
        frequencies = _frequencies(panel_delays)
        codes = {'monthly': 'M', 'quarterly': 'Q', 'annual': 'Y'}
        expected = {key: _days_after_start(date, frequencies[key], codes[target[key[1]]]) for key, date in last.items()}
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
        frequencies = _frequencies(panel_delays)
        expected = {key: from_days(_days_after_start(date, frequencies[key], 'M'))
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

        frequencies = _frequencies(series_delays)
        expected = {column: _days_after_end(date, frequencies[column])
                    for column, date in _last_observations(irregular_index_timeseries).items()}
        assert result['delay'].to_dict() == expected

    def test_quarterly_target_counted_from_the_start(self, irregular_index_timeseries, series_delays):
        """Quarterly target from the start of the quarter containing the last day of each observation period."""
        result = calculate_applicable_delay(series_delays, 'start', 'Q')

        frequencies = _frequencies(series_delays)
        expected = {column: _days_after_start(date, frequencies[column], 'Q')
                    for column, date in _last_observations(irregular_index_timeseries).items()}
        assert result['delay'].to_dict() == expected

    def test_every_column_is_counted_once(self, series_delays):
        """A first download gives a single observation per column."""
        result = calculate_applicable_delay(series_delays, 'end', 'M')

        assert set(result['n_observations']) == {1}
