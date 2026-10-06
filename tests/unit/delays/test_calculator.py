"""Unit tests for ``tsforecast.delays.calculator``.

Covers the single public symbol of the module, ``calculate_applicable_delay``
(parameters ``publication_delays``, ``reference_point``, ``frequency``, ``unit``,
``indicators``, ``aggregate_by_panel`` and ``aggregation_method``), through its
public API only: the private helpers ``_validate_columns``,
``_convert_to_target_frequency_and_reference``, ``_calculate_converted_delay``,
``_convert_delay_unit`` and ``_aggregate_delays`` are exercised through it.

The input contract is the output of ``compare_and_detect_delays``: the main
fixtures are built with it (class ``TestContractWithDataManager``), so that the
two functions are tested against each other. The other classes feed small frames
written by hand, in the same format (frequency literals, period end exclusive,
last index level = indicator), to control every gold value. These are computed
from the calendar, never copied from the output of the code:

* the download date is rebuilt as ``reference date + delay``;
* the delay is then counted again from the start or the end (exclusive) of the
  target period, the one that contains the observation date, or, when the target
  frequency is higher, the sub-period of the source period that contains it.

The realistic scenario (notebook 3 datasets) lives in
``tests/integration/delays/test_applicable_delay_realistic.py``.

Anomalies found while writing these tests are registered in
``tests/ANOMALIES.md`` (``ANO-DELAYS-002``, ``-003``, ``-011`` to ``-015``, all fixed).
"""
# Modules de base
import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

# Fonction à tester
from tsforecast.delays import calculator
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays

TS = pd.Timestamp

# Colonnes requises en entrée (ordre du message d'erreur)
REQUIRED_COLUMNS = [
    'observation_date', 'download_date', 'frequency', 'period_start',
    'period_end', 'reference_point', 'delay', 'unit',
]

# Colonnes de sortie, dans l'ordre
OUTPUT_COLUMNS = ['delay', 'unit', 'frequency', 'reference_point', 'n_observations', 'aggregation_method']

# Code pandas de chaque unité de délai produite par compare_and_detect_delays
_PANDAS_UNITS = {'day': 'D', 'second': 's', 'microsecond': 'us'}


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _row(observation, start, end, delay, *, reference_point='end', unit='day', frequency='monthly') -> dict:
    """Build one publication-delay row, in the format of ``compare_and_detect_delays``.

    Args:
        observation: Date of the observation.
        start: Start of its period.
        end: Exclusive end of its period (first instant of the next period).
        delay: Delay counted from ``reference_point``, in ``unit``.
        reference_point: Reference point of the delay (``'start'`` or ``'end'``).
        unit: Unit of the delay.
        frequency: Frequency of the (entity, indicator) couple, as a literal.

    Returns:
        A dictionary holding the required columns; the download date is rebuilt as
        ``reference date + delay`` (``NaT`` for a ``NaN`` delay).
    """
    reference = TS(end if reference_point == 'end' else start)
    download = reference + pd.Timedelta(delay, unit=_PANDAS_UNITS.get(unit, unit))
    return {
        'observation_date': TS(observation), 'download_date': download, 'frequency': frequency,
        'period_start': TS(start), 'period_end': TS(end), 'reference_point': reference_point,
        'delay': delay, 'unit': unit,
    }


def _monthly(month: str = '2023-12', delay=14, **kwargs) -> dict:
    """Build the row of a monthly observation made on the 15th of ``month`` (``'YYYY-MM'``)."""
    period = pd.Period(month, freq='M')
    return _row(period.start_time + pd.Timedelta(days=14), period.start_time, (period + 1).start_time, delay,
                frequency=kwargs.pop('frequency', 'monthly'), **kwargs)


def _quarterly(observation: str = '2024-03-15', delay=44, **kwargs) -> dict:
    """Build the row of a Q1 2024 observation: period ``[2024-01-01, 2024-04-01)``, download on 2024-05-15 by default."""
    return _row(observation, '2024-01-01', '2024-04-01', delay, frequency='quarterly', **kwargs)


def _annual(observation: str = '2023-06-15', delay=74, **kwargs) -> dict:
    """Build the row of a 2023 observation: period ``[2023-01-01, 2024-01-01)``, download on 2024-03-15 by default."""
    return _row(observation, '2023-01-01', '2024-01-01', delay, frequency='annual', **kwargs)


def _frame(rows: list, index=None, names=('indicator',)) -> pd.DataFrame:
    """Build a publication-delays frame from rows.

    Args:
        rows: Dictionaries returned by :func:`_row` (an empty list gives an empty frame
            with the required columns).
        index: One key per row (a tuple for a ``MultiIndex``); defaults to ``'PIB'``.
        names: Names of the index levels, the last one being the indicator.

    Returns:
        The frame, indexed by ``index``.
    """
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS) if not rows else pd.DataFrame(rows)
    keys = list(index) if index is not None else ['PIB'] * len(rows)
    if len(names) == 1:
        frame.index = pd.Index(keys, name=names[0])
    else:
        frame.index = pd.MultiIndex.from_tuples(keys, names=names)
    return frame


def _detect(*args, **kwargs) -> pd.DataFrame:
    """Call ``compare_and_detect_delays`` silencing the index-replacement and undetectable-frequency warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compare_and_detect_delays(*args, **kwargs)


def _delay(frame: pd.DataFrame, reference_point: str, frequency, **kwargs) -> list:
    """Return the ``delay`` column of ``calculate_applicable_delay`` as a list."""
    return calculate_applicable_delay(frame, reference_point, frequency, **kwargs)['delay'].tolist()


# =============================================================================
# Jeux issus de compare_and_detect_delays (contrat entre les deux fonctions)
# =============================================================================
@pytest.fixture
def monthly_publication_delays() -> pd.DataFrame:
    """Monthly delays of a two-country panel, detected by ``compare_and_detect_delays``.

    ``new`` holds ``PIB`` and ``inflation`` from January to June 2023 for France
    and Germany; ``existing`` lacks the last months, so these are the observations
    detected, downloaded on 2023-08-10 and counted from the period start:

    * (France, PIB): April, May, June, i.e. 131, 101 and 70 days
      (April 1st + 131 d = August 10th: 30 + 31 + 30 + 31 = 122 d to August 1st, then 9 d);
    * (France, inflation): May, June: 101 and 70 days;
    * (Germany, PIB): May, June: 101 and 70 days;
    * (Germany, inflation): June: 70 days.
    """
    months = pd.date_range('2023-01-01', periods=6, freq='MS')
    index = pd.MultiIndex.from_product([['France', 'Germany'], months], names=['country', 'date'])
    new = pd.DataFrame({'PIB': np.arange(1.0, 13.0), 'inflation': np.arange(101.0, 113.0)}, index=index)
    existing = new.copy()
    first_new = {('France', 'PIB'): '2023-04-01', ('France', 'inflation'): '2023-05-01',
                 ('Germany', 'PIB'): '2023-05-01', ('Germany', 'inflation'): '2023-06-01'}
    for (country, column), date in first_new.items():
        existing.loc[(country, slice(TS(date), None)), column] = np.nan

    return _detect(new, existing, download_date='2023-08-10', reference_point='start')


@pytest.fixture
def quarterly_publication_delays() -> pd.DataFrame:
    """Quarterly delays of a two-country panel, detected by ``compare_and_detect_delays``.

    Observations are dated on the first day of their quarter (``QS``), downloaded on
    2023-08-10 and counted from the period start: (France, PIB) Q1 2023 and Q2 2023
    (221 and 131 days, January 1st + 221 d = August 10th: 212 d to August 1st, then 9 d),
    (Germany, PIB) Q2 2023 (131 days).
    """
    quarters = pd.date_range('2022-01-01', periods=6, freq='QS')
    index = pd.MultiIndex.from_product([['France', 'Germany'], quarters], names=['country', 'date'])
    new = pd.DataFrame({'PIB': np.arange(1.0, 13.0)}, index=index)
    existing = new.copy()
    existing.loc[('France', slice(TS('2023-01-01'), None)), 'PIB'] = np.nan
    existing.loc[('Germany', slice(TS('2023-04-01'), None)), 'PIB'] = np.nan

    return _detect(new, existing, download_date='2023-08-10', reference_point='start')


@pytest.fixture
def simple_time_series_delays() -> pd.DataFrame:
    """Delays of a plain time series (no panel), from a first download on 2024-01-15.

    ``PIB`` and ``inflation`` are observed up to December 2023 (delay from the period
    start: 45 days, December 1st to January 15th), ``unemployment`` up to November 2023
    (75 days).
    """
    months = pd.date_range('2023-01-01', periods=12, freq='MS')
    data = pd.DataFrame({'PIB': np.arange(12.0), 'inflation': np.arange(12.0) + 2.5, 'unemployment': np.arange(12.0)},
                        index=months)
    data.loc[TS('2023-12-01'), 'unemployment'] = np.nan

    return _detect(data, None, download_date='2024-01-15', reference_point='start')


@pytest.fixture
def delays_in_seconds() -> pd.DataFrame:
    """One monthly ``PIB`` delay in seconds: December 2023, downloaded 2024-01-15 12:30:45, from the period end.

    Gold: 14 d + 12 h + 30 min + 45 s = 1 209 600 + 43 200 + 1 800 + 45 = 1 254 645 s.
    """
    data = pd.DataFrame({'PIB': np.arange(12.0)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))

    return _detect(data, None, download_date='2024-01-15 12:30:45', reference_point='end', delay_unit='s')


@pytest.fixture
def multi_observation_delays() -> pd.DataFrame:
    """Several observations of ``PIB`` per country, same frequency and reference as the target.

    The delays are kept as they are by a ``'M'`` / ``'end'`` conversion: France 14, 13, 15, 14 and
    Germany 19, 17 (hence 15.33 mean, 14.5 median, 19 max, 13 min and 92 sum over the six rows).
    """
    rows = [_monthly('2023-09', 14), _monthly('2023-10', 13), _monthly('2023-11', 15), _monthly('2023-12', 14),
            _monthly('2023-11', 19), _monthly('2023-12', 17)]
    keys = [('France', 'PIB')] * 4 + [('Germany', 'PIB')] * 2

    return _frame(rows, keys, names=('country', 'indicator'))


# =============================================================================
# Contrat de la sortie
# =============================================================================
class TestOutputContract:
    """Shape, order, labels and types of the returned frame."""

    def test_columns_and_their_order(self, monthly_publication_delays):
        """The output exposes the six documented columns, in a fixed order."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M')

        assert list(result.columns) == OUTPUT_COLUMNS

    def test_columns_order_is_independent_of_the_options(self, monthly_publication_delays):
        """The same columns come out with panel aggregation, a unit and a custom aggregation."""
        result = calculate_applicable_delay(monthly_publication_delays, 'start', 'Q', unit='h',
                                            aggregate_by_panel=True, aggregation_method='max')

        assert list(result.columns) == OUTPUT_COLUMNS

    def test_default_aggregation_is_indexed_by_indicator(self, monthly_publication_delays):
        """Without panel aggregation the result is indexed by the indicator level only (named ``column``)."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M')

        assert list(result.index) == ['PIB', 'inflation'] and result.index.name == 'column'

    def test_panel_aggregation_keeps_all_the_index_level_names(self, monthly_publication_delays):
        """With ``aggregate_by_panel`` the result keeps the entity levels, then the indicator."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregate_by_panel=True)

        assert list(result.index.names) == ['country', 'column']

    def test_panel_aggregation_gives_one_row_per_couple(self, monthly_publication_delays):
        """One row per (country, indicator) couple, sorted."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregate_by_panel=True)

        assert list(result.index) == [('France', 'PIB'), ('France', 'inflation'),
                                      ('Germany', 'PIB'), ('Germany', 'inflation')]

    def test_time_series_without_panel_is_indexed_by_the_indicator(self, simple_time_series_delays):
        """A series without entity level gives one row per indicator, whatever the panel option."""
        result = calculate_applicable_delay(simple_time_series_delays, 'end', 'M', aggregate_by_panel=True)

        assert list(result.index) == ['PIB', 'inflation', 'unemployment']

    def test_delay_is_float_and_count_is_integer(self, monthly_publication_delays):
        """The aggregated delay is a float, the number of observations an integer."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M')

        assert (result['delay'].dtype, result['n_observations'].dtype) == (np.dtype('float64'), np.dtype('int64'))

    @pytest.mark.parametrize('frequency', ['M', 'monthly'], ids=['code', 'literal'])
    def test_frequency_is_reported_as_given(self, monthly_publication_delays, frequency):
        """The ``frequency`` column holds the target frequency exactly as passed (code or literal)."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', frequency)

        assert result['frequency'].unique().tolist() == [frequency]

    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_reference_point_is_the_target_one(self, monthly_publication_delays, reference_point):
        """The ``reference_point`` column holds the target reference point, not the one of the input."""
        result = calculate_applicable_delay(monthly_publication_delays, reference_point, 'M')

        assert result['reference_point'].unique().tolist() == [reference_point]

    def test_unit_label_is_kept_without_target_unit(self, simple_time_series_delays):
        """Without ``unit`` the label of the input is unchanged (``'day'``, not ``'D'``)."""
        result = calculate_applicable_delay(simple_time_series_delays, 'end', 'M')

        assert result['unit'].unique().tolist() == ['day']

    def test_unit_label_is_the_code_with_a_target_unit(self, simple_time_series_delays):
        """With ``unit`` the label is the duration code (``'D'`` for ``'day'``)."""
        result = calculate_applicable_delay(simple_time_series_delays, 'end', 'M', unit='day')

        assert result['unit'].unique().tolist() == ['D']

    def test_aggregation_method_is_reported_by_name(self, monthly_publication_delays):
        """The ``aggregation_method`` column holds the method name."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregation_method='mean')

        assert result['aggregation_method'].unique().tolist() == ['mean']

    def test_input_is_not_modified(self, monthly_publication_delays):
        """The input frame is left as it was (the conversion works on a copy)."""
        before = monthly_publication_delays.copy(deep=True)

        calculate_applicable_delay(monthly_publication_delays, 'end', 'Q', unit='h')

        pd.testing.assert_frame_equal(monthly_publication_delays, before)

    def test_calls_are_repeatable(self, monthly_publication_delays):
        """Two calls on the same input give the same result."""
        first = calculate_applicable_delay(monthly_publication_delays, 'end', 'Q', aggregate_by_panel=True)
        second = calculate_applicable_delay(monthly_publication_delays, 'end', 'Q', aggregate_by_panel=True)

        pd.testing.assert_frame_equal(first, second)

    def test_docstring_examples_run(self):
        """The examples of the module docstrings (public function and helpers) all pass."""
        results = doctest.testmod(calculator, optionflags=doctest.ELLIPSIS)

        assert results.attempted > 0 and results.failed == 0


# =============================================================================
# Colonnes requises
# =============================================================================
class TestRequiredColumns:
    """Validation of the columns of ``publication_delays``."""

    @pytest.mark.parametrize('column', REQUIRED_COLUMNS)
    def test_each_missing_column_is_named(self, monthly_publication_delays, column):
        """Dropping any required column raises a ``ValueError`` that names it."""
        delays = monthly_publication_delays.drop(columns=[column])

        with pytest.raises(ValueError, match=f"Missing required columns.*'{column}'"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_all_missing_columns_are_listed_in_order(self, monthly_publication_delays):
        """Several missing columns are all listed, in the order of the required columns."""
        delays = monthly_publication_delays.drop(columns=['unit', 'frequency'])

        with pytest.raises(ValueError, match=r"publication_delays DataFrame: \['frequency', 'unit'\]\."):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_message_gives_the_required_columns(self, monthly_publication_delays):
        """The message recalls the full list of required columns."""
        delays = monthly_publication_delays.drop(columns=['unit'])

        with pytest.raises(ValueError, match=r"Required columns are: \['observation_date', 'download_date'"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_former_release_delay_name_is_rejected(self, monthly_publication_delays):
        """A frame still using ``release_delay`` (renamed ``delay``) is rejected, naming ``delay``."""
        delays = monthly_publication_delays.rename(columns={'delay': 'release_delay'})

        with pytest.raises(ValueError, match=r"Missing required columns.*\['delay'\]"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_frame_without_any_required_column_is_rejected(self):
        """A frame holding none of the required columns lists all of them."""
        delays = pd.DataFrame({'x': [1]}, index=pd.Index(['PIB'], name='indicator'))

        with pytest.raises(ValueError, match="Missing required columns.*'observation_date'.*'unit'"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_extra_columns_are_ignored(self, monthly_publication_delays):
        """Extra columns (``has_changes`` of ``compare_and_detect_delays``, a comment) do not disturb the result."""
        delays = monthly_publication_delays.assign(comment='x')

        assert _delay(delays, 'end', 'M') == _delay(monthly_publication_delays.drop(columns=['has_changes']), 'end', 'M')


# =============================================================================
# Point de référence, à fréquence inchangée
# =============================================================================
class TestReferencePoint:
    """Moving the reference of a delay between the start and the end of the same period."""

    @pytest.mark.parametrize(
        ('target', 'expected'),
        [('end', 14), ('start', 45)],
        ids=['end-to-end', 'end-to-start'],
    )
    def test_input_reference_end(self, target, expected):
        """Gold values: download on 2024-01-15 = January 1st + 14 d; counted from December 1st: 31 + 14 = 45 d."""
        delays = _frame([_monthly('2023-12', 14, reference_point='end')])

        assert _delay(delays, target, 'M') == [expected]

    @pytest.mark.parametrize(
        ('target', 'expected'),
        [('start', 45), ('end', 14)],
        ids=['start-to-start', 'start-to-end'],
    )
    def test_input_reference_start(self, target, expected):
        """The download date is rebuilt from the period start when the input counts from it (45 d from December 1st)."""
        delays = _frame([_monthly('2023-12', 45, reference_point='start')])

        assert _delay(delays, target, 'M') == [expected]

    def test_rows_with_different_input_references_are_homogenised(self):
        """Two rows of the same publication, one counted from each bound, give the same delay."""
        delays = _frame([_monthly('2023-12', 14, reference_point='end'), _monthly('2023-12', 45, reference_point='start')])

        assert _delay(delays, 'end', 'M', aggregation_method='max') == [14.0]

    def test_start_delay_is_longer_than_end_delay_by_the_period_length(self, monthly_publication_delays):
        """Counting from the start adds the period length: 30, 31 and 30 days for April, May and June."""
        start = calculate_applicable_delay(monthly_publication_delays, 'start', 'M', aggregate_by_panel=True,
                                           aggregation_method='min')
        end = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregate_by_panel=True,
                                         aggregation_method='min')

        # Valeur d'or : le plus petit délai de chaque couple est celui de juin (30 jours)
        assert (start['delay'] - end['delay']).unique().tolist() == [30.0]

    @pytest.mark.parametrize('reference_point', ['middle', 'START', '', None, 1])
    def test_invalid_reference_point_raises(self, monthly_publication_delays, reference_point):
        """Anything but ``'start'`` / ``'end'`` is refused."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            calculate_applicable_delay(monthly_publication_delays, reference_point, 'M')


# =============================================================================
# Conversions de fréquence
# =============================================================================
class TestFrequencyConversion:
    """Delays recomputed for another target frequency, hand-computed gold values."""

    @pytest.mark.parametrize(
        ('observation', 'target', 'expected'),
        [
            ('2024-01-15', 'start', 135), ('2024-01-15', 'end', 104),
            ('2024-02-15', 'start', 104), ('2024-02-15', 'end', 75),
            ('2024-03-15', 'start', 75), ('2024-03-15', 'end', 44),
        ],
        ids=['jan-start', 'jan-end', 'feb-start', 'feb-end', 'mar-start', 'mar-end'],
    )
    def test_quarterly_to_monthly_follows_the_observation_month(self, observation, target, expected):
        """Q1 2024 observation downloaded on 2024-05-15 (April 1st + 44 d): the target month is the one of the observation.

        Valeurs d'or, 2024 étant bissextile : 1er janv. -> 15 mai = 31 + 29 + 31 + 30 + 14 = 135 j ;
        1er févr. -> 104 j ; 1er mars -> 75 j ; 1er avril -> 44 j (fin de janvier = 1er février, etc.).
        """
        delays = _frame([_quarterly(observation)])

        assert _delay(delays, target, 'M') == [expected]

    @pytest.mark.parametrize(
        ('target', 'expected'),
        [('start', 135), ('end', 44)],
        ids=['start', 'end'],
    )
    def test_quarterly_to_quarterly(self, target, expected):
        """Same frequency: 44 d from April 1st, hence 135 d from January 1st (91 d of quarter length)."""
        delays = _frame([_quarterly('2024-03-15')])

        assert _delay(delays, target, 'Q') == [expected]

    @pytest.mark.parametrize(
        ('month', 'target', 'expected'),
        [
            ('2023-10', 'start', 45), ('2023-10', 'end', -47),
            ('2023-11', 'start', 75), ('2023-11', 'end', -17),
            ('2023-12', 'start', 106), ('2023-12', 'end', 14),
        ],
        ids=['oct-start', 'oct-end', 'nov-start', 'nov-end', 'dec-start', 'dec-end'],
    )
    def test_monthly_to_quarterly_counts_from_the_quarter_bounds(self, month, target, expected):
        """Each month of Q4 2023, published 14 d after its end, recounted over the quarter ``[Oct 1st, Jan 1st)``.

        Valeurs d'or : octobre -> téléchargé le 15 nov. : 45 j depuis le 1er oct. (31 + 14), -47 j depuis la fin du
        trimestre (15 nov. -> 1er janv.) ; novembre -> 15 déc. : 75 j (31 + 30 + 14) et -17 j ; décembre -> 15 janv. :
        106 j (31 + 30 + 31 + 14) et 14 j. Un mois publié avant la fin du trimestre donne un délai négatif.
        """
        delays = _frame([_monthly(month, 14)])

        assert _delay(delays, target, 'Q') == [expected]

    @pytest.mark.parametrize(
        ('target_frequency', 'target', 'expected'),
        [
            ('M', 'start', 288), ('M', 'end', 258),
            ('Q', 'start', 349), ('Q', 'end', 258),
            ('Y', 'start', 439), ('Y', 'end', 74),
        ],
        ids=['M-start', 'M-end', 'Q-start', 'Q-end', 'Y-start', 'Y-end'],
    )
    def test_annual_observation_converted(self, target_frequency, target, expected):
        """2023 observation of June 15th, downloaded 2024-03-15 (January 1st 2024 + 74 d: 31 + 29 + 14).

        Valeurs d'or : mois de juin -> début 1er juin 2023 : 274 + 14 = 288 j (de juin 2023 à fin février 2024 :
        30 + 31 + 31 + 30 + 31 + 30 + 31 + 31 + 29 = 274), fin 1er juillet : 258 j ; trimestre T2 -> début 1er avril :
        258 + 91 = 349 j ; année -> début 1er janv. 2023 : 365 + 74 = 439 j, fin : 74 j.
        """
        delays = _frame([_annual('2023-06-15')])

        assert _delay(delays, target, target_frequency) == [expected]

    @pytest.mark.parametrize(
        ('target_frequency', 'target', 'expected'),
        [('D', 'start', 31), ('D', 'end', 30), ('W', 'start', 35), ('W', 'end', 28)],
        ids=['D-start', 'D-end', 'W-start', 'W-end'],
    )
    def test_monthly_to_daily_and_weekly(self, target_frequency, target, expected):
        """December 2023 observation of the 15th (a Friday), downloaded 2024-01-15.

        Valeurs d'or : jour -> début le 15 déc. = 31 j, fin le 16 déc. = 30 j ; semaine du lundi 11 au lundi 18 déc. ->
        début : 21 + 14 = 35 j, fin : 14 + 14 = 28 j.
        """
        delays = _frame([_monthly('2023-12', 14)])

        assert _delay(delays, target, target_frequency) == [expected]

    @pytest.mark.parametrize(
        ('target_frequency', 'target', 'expected'),
        [('W', 'start', 9), ('W', 'end', 2), ('D', 'start', 5), ('D', 'end', 4)],
        ids=['W-start', 'W-end', 'D-start', 'D-end'],
    )
    def test_weekly_observation(self, target_frequency, target, expected):
        """Week of Monday 2023-12-11, observed on the 15th, published 2 d after its (exclusive) end: download 2023-12-20."""
        delays = _frame([_row('2023-12-15', '2023-12-11', '2023-12-18', 2, frequency='weekly')])

        assert _delay(delays, target, target_frequency) == [expected]

    @pytest.mark.parametrize(('target', 'expected'), [('start', 7), ('end', 0)], ids=['start', 'end'])
    def test_daily_to_weekly(self, target, expected):
        """Day of Friday 2023-12-15 downloaded 2 d after its end (December 18th), recounted over its week.

        Valeurs d'or : semaine du lundi 11 au lundi 18 déc. -> début : 18 - 11 = 7 j, fin : 18 - 18 = 0 j.
        """
        delays = _frame([_row('2023-12-15', '2023-12-15', '2023-12-16', 2, frequency='daily')])

        assert _delay(delays, target, 'W') == [expected]

    @pytest.mark.parametrize(
        ('spelling', 'literal'),
        [('M', 'monthly'), ('Q', 'quarterly'), ('Y', 'annual'), ('D', 'daily'), ('W', 'weekly')],
        ids=['M', 'Q', 'Y', 'D', 'W'],
    )
    def test_code_and_literal_spellings_are_equivalent(self, spelling, literal):
        """A frequency code and its literal name give the same delay."""
        delays = _frame([_monthly('2023-12', 14)])

        assert _delay(delays, 'start', spelling) == _delay(delays, 'start', literal)

    @pytest.mark.parametrize('source', ['M', 'monthly'], ids=['code', 'literal'])
    def test_source_frequency_may_be_a_code_or_a_literal(self, source):
        """The ``frequency`` column of the input may hold codes as well as literals."""
        delays = _frame([_monthly('2023-12', 14, frequency=source)])

        assert _delay(delays, 'start', 'Q') == [106]

    def test_alias_a_is_not_a_supported_frequency(self):
        """The pandas alias ``'A'`` was deliberately dropped (consolidation of ``parse_frequency``)."""
        with pytest.raises(ValueError, match="Unsupported frequency: A"):
            calculate_applicable_delay(_frame([_annual()]), 'end', 'A')

    @pytest.mark.parametrize('frequency', ['fortnightly', 'zz', ''], ids=['unknown-name', 'unknown-code', 'empty'])
    def test_unsupported_target_frequency_raises(self, frequency):
        """A target frequency the normalizer does not know is refused."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            calculate_applicable_delay(_frame([_monthly()]), 'end', frequency)

    def test_unsupported_source_frequency_raises(self):
        """A frequency the normalizer does not know in the input frame is refused."""
        delays = _frame([_monthly(frequency='zz')])

        with pytest.raises(ValueError, match="Unsupported frequency: zz"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_observation_outside_its_period_falls_back_on_the_last_subperiod(self):
        """An observation date outside the source period selects the last sub-period (documented edge case).

        Q1 2024 row dated 2024-06-15 (after its period): the last month of the quarter, March, is used, hence the
        delays of the March observation (75 d from March 1st, 44 d from April 1st).
        """
        delays = _frame([_quarterly('2024-06-15')])

        assert (_delay(delays, 'start', 'M'), _delay(delays, 'end', 'M')) == ([75], [44])

    def test_observation_before_its_period_falls_back_on_the_last_subperiod(self):
        """Same fallback for an observation date before the source period."""
        delays = _frame([_quarterly('2023-06-15')])

        assert _delay(delays, 'start', 'M') == [75]

    def test_quarterly_dated_at_period_end_maps_onto_the_last_month(self):
        """A quarter labelled on its last day (``QE``) is converted to the last month of the quarter."""
        delays = _frame([_quarterly('2024-03-31')])

        # Valeur d'or : mars 2024 -> début le 1er mars : 75 j
        assert _delay(delays, 'start', 'M') == [75]

    def test_quarterly_dated_at_period_start_maps_onto_the_first_month(self):
        """A quarter labelled on its first day (``QS``) is converted to the first month of the quarter."""
        delays = _frame([_quarterly('2024-01-01')])

        # Valeur d'or : janvier 2024 -> début le 1er janv. : 135 j
        assert _delay(delays, 'start', 'M') == [135]

    def test_mixed_source_frequencies_are_converted_row_by_row(self):
        """Rows of different source frequencies for the same indicator are each converted to the target one."""
        delays = _frame([_quarterly('2024-03-15'), _monthly('2024-03', 14)])

        # Valeurs d'or : T1 -> mars : 75 j ; mensuel de mars (téléchargé le 15 avril) -> 1er mars : 45 j ; médiane 60
        assert _delay(delays, 'start', 'M') == [60.0]


# =============================================================================
# Méthodes d'agrégation
# =============================================================================
class TestAggregation:
    """Aggregation of the converted delays by indicator or by (entity, indicator)."""

    @pytest.mark.parametrize(
        ('method', 'expected'),
        [('mean', 92 / 6), ('median', 14.5), ('max', 19), ('min', 13), ('sum', 92)],
    )
    def test_method_by_indicator(self, multi_observation_delays, method, expected):
        """Valeurs d'or sur les six délais 14, 13, 15, 14, 19, 17 : somme 92, moyenne 15,33, médiane (14 + 15) / 2."""
        assert _delay(multi_observation_delays, 'end', 'M', aggregation_method=method) == pytest.approx([expected])

    @pytest.mark.parametrize(
        ('method', 'france', 'germany'),
        [('mean', 14, 18), ('median', 14, 18), ('max', 15, 19), ('min', 13, 17), ('sum', 56, 36)],
    )
    def test_method_by_panel(self, multi_observation_delays, method, france, germany):
        """France 14, 13, 15, 14 and Germany 19, 17, aggregated per entity."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregate_by_panel=True,
                                            aggregation_method=method)

        assert result['delay'].to_dict() == {('France', 'PIB'): france, ('Germany', 'PIB'): germany}

    def test_default_method_is_the_median(self, multi_observation_delays):
        """Without ``aggregation_method`` the median is used."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M')

        assert (result['delay'].tolist(), result['aggregation_method'].tolist()) == ([14.5], ['median'])

    def test_callable_method(self, multi_observation_delays):
        """A callable receives the delays of each group; its name is reported.

        Valeur d'or : quantile 0,75 de 13, 14, 14, 15, 17, 19 = 15 + 0,75 * (17 - 15) = 16,5.
        """
        def third_quartile(delays):
            return delays.quantile(0.75)

        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregation_method=third_quartile)

        assert (result['delay'].tolist(), result['aggregation_method'].tolist()) == ([16.5], ['third_quartile'])

    def test_lambda_method_is_reported_as_lambda(self, multi_observation_delays):
        """An anonymous function is reported by its ``__name__``."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregation_method=lambda x: x.max())

        assert result['aggregation_method'].tolist() == ['<lambda>']

    def test_callable_method_by_panel(self, multi_observation_delays):
        """A callable is applied per (entity, indicator) group: the range of the delays here (15 - 13 and 19 - 17)."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregate_by_panel=True,
                                            aggregation_method=lambda x: x.max() - x.min())

        assert result['delay'].tolist() == [2.0, 2.0]

    def test_number_of_observations_by_indicator(self, multi_observation_delays):
        """All six rows of the indicator are counted."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M')

        assert result['n_observations'].tolist() == [6]

    def test_number_of_observations_by_panel(self, multi_observation_delays):
        """Four observations for France, two for Germany."""
        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregate_by_panel=True)

        assert result['n_observations'].tolist() == [4, 2]

    def test_aggregation_runs_after_the_conversion(self, monthly_publication_delays):
        """The aggregated value is computed on the converted delays, not on the input ones.

        (France, PIB) delays 131, 101 and 70 d from the start of April, May and June, converted
        to quarterly from the start: the three belong to Q2, i.e. 131 d each.
        """
        result = calculate_applicable_delay(monthly_publication_delays, 'start', 'Q', aggregate_by_panel=True,
                                            aggregation_method='min')

        assert result.loc[('France', 'PIB'), 'delay'] == 131

    @pytest.mark.parametrize('method', ['nope', 'Mean', ''], ids=['unknown', 'wrong-case', 'empty'])
    def test_unknown_method_name_raises_a_value_error_naming_the_argument(self, multi_observation_delays, method):
        """A name that is not a pandas aggregation is a ``ValueError`` naming ``aggregation_method`` (ANO-DELAYS-015)."""
        with pytest.raises(ValueError, match=f"Unsupported aggregation_method {method!r}"):
            calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregation_method=method)

    @pytest.mark.parametrize('method', [3, None, ['mean']], ids=['int', 'none', 'list'])
    def test_non_callable_method_raises_a_type_error_naming_the_argument(self, multi_observation_delays, method):
        """A method that is neither a name nor a callable is a ``TypeError`` naming ``aggregation_method``."""
        with pytest.raises(TypeError, match="'aggregation_method' should be a string or a callable"):
            calculate_applicable_delay(multi_observation_delays, 'end', 'M', aggregation_method=method)

    def test_method_is_validated_before_the_computation(self, multi_observation_delays):
        """The method is checked first: a wrong method is reported even when the frame is otherwise rejected."""
        with pytest.raises(ValueError, match="Unsupported aggregation_method"):
            calculate_applicable_delay(multi_observation_delays, 'end', 'M', indicators=['PIB'],
                                       aggregation_method='nope')

    def test_callable_without_a_name_is_reported_by_its_type(self, multi_observation_delays):
        """A callable object without ``__name__`` (a ``functools.partial``) is reported by its type name."""
        import functools

        result = calculate_applicable_delay(multi_observation_delays, 'end', 'M',
                                            aggregation_method=functools.partial(pd.Series.quantile, q=0.25))

        assert result['aggregation_method'].tolist() == ['partial']


# =============================================================================
# Unité de sortie
# =============================================================================
class TestTargetUnit:
    """Conversion of the delays to the ``unit`` argument (ceiling rounding)."""

    @pytest.mark.parametrize(
        ('unit', 'code', 'expected'),
        [
            ('D', 'D', 14), ('day', 'D', 14),
            ('h', 'h', 336), ('hour', 'h', 336),
            ('min', 'min', 20160), ('minute', 'min', 20160),
            ('W', 'W', 2), ('week', 'W', 2),
            ('s', 's', 1_209_600), ('second', 's', 1_209_600),
            ('ms', 'ms', 1_209_600_000), ('millisecond', 'ms', 1_209_600_000),
            ('us', 'us', 1_209_600_000_000), ('microsecond', 'us', 1_209_600_000_000),
        ],
    )
    def test_conversion_of_a_whole_number_of_days(self, unit, code, expected):
        """14 d = 336 h = 20 160 min = 2 weeks = 1 209 600 s, and the label is the duration code."""
        result = calculate_applicable_delay(_frame([_monthly('2023-12', 14)]), 'end', 'M', unit=unit)

        assert (result['delay'].tolist(), result['unit'].tolist()) == ([expected], [code])

    def test_ceiling_rounding_to_a_coarser_unit(self):
        """15 days are 2.14 weeks, rounded up to 3."""
        assert _delay(_frame([_monthly('2023-12', 15)]), 'end', 'M', unit='W') == [3]

    def test_ceiling_rounding_of_seconds_to_days(self, delays_in_seconds):
        """1 254 645 s = 14.52 d, rounded up to 15 d."""
        assert _delay(delays_in_seconds, 'end', 'M', unit='D') == [15]

    def test_ceiling_rounding_of_seconds_to_hours(self, delays_in_seconds):
        """1 254 645 s = 348.51 h, rounded up to 349 h."""
        assert _delay(delays_in_seconds, 'end', 'M', unit='h') == [349]

    def test_seconds_delay_counted_from_the_start(self, delays_in_seconds):
        """Counted from December 1st (31 d = 2 678 400 s earlier): 3 933 045 s, i.e. 45.52 d, rounded up to 46."""
        assert _delay(delays_in_seconds, 'start', 'M', unit='D') == [46]

    def test_seconds_delay_without_target_unit(self, delays_in_seconds):
        """Without ``unit`` the delay stays in seconds, the (whole) value is kept."""
        result = calculate_applicable_delay(delays_in_seconds, 'end', 'M')

        assert (result['delay'].tolist(), result['unit'].tolist()) == ([1_254_645], ['second'])

    def test_negative_delay_is_rounded_up_towards_zero(self):
        """-129 600 s = -1.5 d, rounded up (ceiling) to -1 d."""
        delays = _frame([_monthly('2023-12', -129_600, unit='second')])

        assert _delay(delays, 'end', 'M', unit='D') == [-1]

    def test_same_unit_is_left_as_it_is(self, simple_time_series_delays):
        """Converting to the unit of the data changes nothing but the label (``'day'`` becomes ``'D'``)."""
        kept = calculate_applicable_delay(simple_time_series_delays, 'end', 'M')
        converted = calculate_applicable_delay(simple_time_series_delays, 'end', 'M', unit='day')

        assert (converted['delay'].tolist(), converted['unit'].unique().tolist()) == (kept['delay'].tolist(), ['D'])

    def test_unit_is_applied_after_the_reference_conversion(self):
        """Hour delay of the start-reference conversion: 45 d = 1 080 h."""
        assert _delay(_frame([_monthly('2023-12', 14)]), 'start', 'M', unit='h') == [1080]

    def test_mixed_input_units_are_converted_with_a_target_unit(self):
        """14 d and 14 * 86 400 s are the same delay: with ``unit='D'`` both give 14 d."""
        delays = _frame([_monthly('2023-12', 14), _monthly('2023-12', 14 * 86_400, unit='second')],
                        [('France', 'PIB'), ('Germany', 'PIB')], names=('country', 'indicator'))

        result = calculate_applicable_delay(delays, 'end', 'M', unit='D', aggregate_by_panel=True)

        assert result['delay'].tolist() == [14, 14]

    @pytest.mark.parametrize('unit', ['days', 'fortnight', 'seconds', 'H'], ids=['plural', 'unknown', 'plural-s', 'upper'])
    def test_unsupported_target_unit_raises(self, unit):
        """A duration the converter does not know (plural forms included) is refused."""
        with pytest.raises(ValueError, match="Unsupported duration"):
            calculate_applicable_delay(_frame([_monthly()]), 'end', 'M', unit=unit)

    def test_plural_unit_in_the_input_raises(self):
        """A plural unit in the ``unit`` column of the input (``'days'``) is refused, as documented."""
        with pytest.raises(ValueError, match="Unsupported duration: days"):
            calculate_applicable_delay(_frame([_monthly(unit='days')]), 'end', 'M')

    def test_mixed_input_units_are_refused_without_target_unit(self):
        """Without ``unit``, rows of one indicator in days and in seconds are refused, naming the group (ANO-DELAYS-002)."""
        delays = _frame([_monthly('2023-12', 14), _monthly('2023-12', 14 * 86_400, unit='second')],
                        [('France', 'GDP'), ('Germany', 'GDP')], names=('country', 'indicator'))

        with pytest.raises(ValueError, match=r"delays of \['GDP'\] are expressed in different units"):
            calculate_applicable_delay(delays, 'end', 'M')

    def test_mixed_input_units_are_accepted_when_aggregated_per_couple(self):
        """Each (entity, indicator) group having its own single unit, the panel aggregation keeps them as they are."""
        delays = _frame([_monthly('2023-12', 14), _monthly('2023-12', 14 * 86_400, unit='second')],
                        [('France', 'GDP'), ('Germany', 'GDP')], names=('country', 'indicator'))

        result = calculate_applicable_delay(delays, 'end', 'M', aggregate_by_panel=True)

        assert (result['delay'].tolist(), result['unit'].tolist()) == ([14, 14 * 86_400], ['day', 'second'])

    def test_day_name_and_code_are_the_same_unit(self):
        """``'day'`` and ``'D'`` are one unit: rows labelled either way are aggregated without a conversion."""
        delays = _frame([_monthly('2023-12', 14, unit='day'), _monthly('2023-12', 16, unit='D')])

        assert _delay(delays, 'end', 'M') == [15]

    @pytest.mark.parametrize(
        ('delay', 'target', 'expected'),
        [(8_183_426_019_070, 'end', 8_183_426_019_070),
         (129_600_000_123_457, 'start', 129_600_000_123_457 + 31 * 86_400 * 10**6),
         (1, 'end', 1)],
        ids=['end-to-end', 'end-to-start', 'one-microsecond'],
    )
    def test_microsecond_delay_is_exact(self, delay, target, expected):
        """A microsecond delay is converted exactly (31 d of December = 31 * 86 400 * 10^6 µs added from the start).

        The conversion works on integer nanoseconds (ANO-DELAYS-012: float seconds used to lose 1 µs).
        """
        delays = _frame([_monthly('2023-12', delay, unit='microsecond')])

        assert _delay(delays, target, 'M') == [expected]


# =============================================================================
# Fréquence cible par indicateur
# =============================================================================
class TestTargetFrequencyDictionary:
    """``frequency`` as a dictionary: one frequency per indicator, or per (entity, indicator) couple."""

    @pytest.fixture
    def gdp_and_cpi(self) -> pd.DataFrame:
        """Quarterly ``GDP`` (Q1 2024, observed in March, download 2024-05-15) and monthly ``CPI`` (March, 2024-04-15)."""
        return _frame([_quarterly('2024-03-15'), _monthly('2024-03', 14)], ['GDP', 'CPI'])

    @pytest.fixture
    def panel_gdp(self) -> pd.DataFrame:
        """Quarterly ``GDP`` of two countries and monthly ``CPI`` of one, all observed in March 2024.

        Download on 2024-05-15 for GDP, on 2024-04-15 for the CPI.
        """
        return _frame([_quarterly('2024-03-15'), _quarterly('2024-03-15'), _monthly('2024-03', 14)],
                      [('FR', 'GDP'), ('DE', 'GDP'), ('FR', 'CPI')], names=('country', 'indicator'))

    def test_each_indicator_gets_its_own_frequency(self, gdp_and_cpi):
        """GDP in monthly: 75 d from March 1st; CPI in quarterly: 105 d from January 1st (31 + 29 + 31 + 14)."""
        result = calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'monthly', 'CPI': 'quarterly'})

        assert result['delay'].to_dict() == {'CPI': 105, 'GDP': 75}

    def test_frequency_column_is_the_one_of_each_indicator(self, gdp_and_cpi):
        """Each row reports the target frequency of its indicator, as given."""
        result = calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'M', 'CPI': 'quarterly'})

        assert result['frequency'].to_dict() == {'CPI': 'quarterly', 'GDP': 'M'}

    def test_same_frequency_for_all_with_a_string(self, gdp_and_cpi):
        """A single string applies to every indicator: 75 d (GDP) and 45 d (CPI, March 1st to April 15th)."""
        assert calculate_applicable_delay(gdp_and_cpi, 'start', 'M')['delay'].to_dict() == {'CPI': 45, 'GDP': 75}

    def test_dictionary_equal_to_the_string_gives_the_same_result(self, gdp_and_cpi):
        """A dictionary repeating one frequency is the same as that string."""
        by_dict = calculate_applicable_delay(gdp_and_cpi, 'end', {'GDP': 'M', 'CPI': 'M'})

        pd.testing.assert_frame_equal(by_dict, calculate_applicable_delay(gdp_and_cpi, 'end', 'M'))

    def test_dictionary_with_extra_indicators_is_accepted(self, gdp_and_cpi):
        """Indicators absent from the data are ignored."""
        result = calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'M', 'CPI': 'M', 'UNEMPLOYMENT': 'Q'})

        assert list(result.index) == ['CPI', 'GDP']

    def test_dictionary_must_cover_only_the_selected_indicators(self, gdp_and_cpi):
        """With ``indicators`` the dictionary has to cover the filtered indicators only."""
        result = calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'M'}, indicators=['GDP'])

        assert result['delay'].to_dict() == {'GDP': 75}

    def test_couple_keys_give_a_frequency_per_entity_and_indicator(self, panel_gdp):
        """French GDP in monthly (75 d from March 1st), German GDP in quarterly (135 d from January 1st), CPI in monthly."""
        result = calculate_applicable_delay(
            panel_gdp, 'start', {('FR', 'GDP'): 'monthly', ('DE', 'GDP'): 'quarterly', ('FR', 'CPI'): 'monthly'},
            aggregate_by_panel=True)

        assert result['delay'].to_dict() == {('DE', 'GDP'): 135, ('FR', 'CPI'): 45, ('FR', 'GDP'): 75}

    def test_couple_keys_are_reported_per_couple(self, panel_gdp):
        """The ``frequency`` column gives the frequency of each couple."""
        result = calculate_applicable_delay(
            panel_gdp, 'start', {('FR', 'GDP'): 'M', ('DE', 'GDP'): 'Q', 'CPI': 'M'}, aggregate_by_panel=True)

        assert result['frequency'].to_dict() == {('DE', 'GDP'): 'Q', ('FR', 'CPI'): 'M', ('FR', 'GDP'): 'M'}

    def test_couple_key_takes_precedence_over_the_indicator_key(self, panel_gdp):
        """The indicator key is the default, a couple key overrides it: GDP in monthly except for Germany."""
        result = calculate_applicable_delay(
            panel_gdp, 'start', {'GDP': 'M', ('DE', 'GDP'): 'Q', 'CPI': 'M'}, aggregate_by_panel=True)

        assert result['delay'].to_dict() == {('DE', 'GDP'): 135, ('FR', 'CPI'): 45, ('FR', 'GDP'): 75}

    def test_couple_keys_of_a_three_level_index(self):
        """With a three-level index the key holds every level: (region, country, indicator)."""
        keys = [('EU', 'France', 'PIB'), ('EU', 'Germany', 'PIB')]
        delays = _frame([_quarterly('2024-03-15'), _quarterly('2024-03-15')], keys,
                        names=('region', 'country', 'indicator'))

        result = calculate_applicable_delay(
            delays, 'start', {('EU', 'France', 'PIB'): 'M', ('EU', 'Germany', 'PIB'): 'Q'}, aggregate_by_panel=True)

        assert result['delay'].tolist() == [75, 135]

    def test_entities_with_different_frequencies_need_the_panel_aggregation(self, panel_gdp):
        """Without ``aggregate_by_panel``, two entities of an indicator cannot have different target frequencies."""
        with pytest.raises(ValueError, match=r"differs between the entities of the indicators \['GDP'\]"):
            calculate_applicable_delay(panel_gdp, 'start', {('FR', 'GDP'): 'M', ('DE', 'GDP'): 'Q', 'CPI': 'M'})

    def test_entities_with_the_same_frequency_may_be_aggregated(self, panel_gdp):
        """Couple keys giving one frequency per indicator are fine without the panel aggregation (spellings aside)."""
        result = calculate_applicable_delay(panel_gdp, 'start', {('FR', 'GDP'): 'M', ('DE', 'GDP'): 'monthly', 'CPI': 'M'})

        assert result['delay'].to_dict() == {'CPI': 45, 'GDP': 75}

    def test_uncovered_couple_is_named(self, panel_gdp):
        """A couple covered by neither its own key nor its indicator key is named in the error."""
        with pytest.raises(ValueError, match=r"\('DE', 'GDP'\)"):
            calculate_applicable_delay(panel_gdp, 'start', {('FR', 'GDP'): 'M', 'CPI': 'M'}, aggregate_by_panel=True)

    def test_dictionary_keyed_by_entity_alone_is_refused(self, monthly_publication_delays):
        """A dictionary keyed by country alone covers neither the couples nor the indicators."""
        with pytest.raises(ValueError, match="does not give a target frequency"):
            calculate_applicable_delay(monthly_publication_delays, 'end', {'France': 'M', 'Germany': 'Q'})

    def test_uncovered_indicator_is_named(self, gdp_and_cpi):
        """An indicator missing from the dictionary is named in the error message (ANO-DELAYS-003)."""
        with pytest.raises(ValueError, match="CPI"):
            calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'M'})

    def test_unsupported_frequency_in_the_dictionary_raises(self, gdp_and_cpi):
        """An unsupported frequency in the dictionary is refused like a string one."""
        with pytest.raises(ValueError, match="Unsupported frequency: zz"):
            calculate_applicable_delay(gdp_and_cpi, 'start', {'GDP': 'M', 'CPI': 'zz'})

    @pytest.mark.parametrize('frequency', [None, 123, ['M'], ('M',), 1.5], ids=['none', 'int', 'list', 'tuple', 'float'])
    def test_invalid_frequency_type_raises(self, monthly_publication_delays, frequency):
        """Anything but a string or a dictionary is a ``TypeError``, naming the received type."""
        with pytest.raises(TypeError, match=f"'frequency' should be a string or a dict, got a {type(frequency).__name__}"):
            calculate_applicable_delay(monthly_publication_delays, 'end', frequency)


# =============================================================================
# Indicateurs et panel
# =============================================================================
class TestIndicatorsAndPanel:
    """Selection of the indicators and aggregation levels."""

    def test_selected_indicators_only(self, monthly_publication_delays):
        """Only the requested indicator is in the result."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators=['PIB'])

        assert list(result.index) == ['PIB']

    def test_selection_restricts_the_observations_counted(self, monthly_publication_delays):
        """The five ``PIB`` observations (France 3, Germany 2) are counted, not the inflation ones."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators=['PIB'])

        assert result['n_observations'].tolist() == [5]

    def test_unknown_indicators_are_ignored_when_one_is_found(self, monthly_publication_delays):
        """A requested indicator absent from the data is ignored if another one is present."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators=['PIB', 'unknown'])

        assert list(result.index) == ['PIB']

    @pytest.mark.parametrize('indicators', [['unknown'], []], ids=['unknown', 'empty-list'])
    def test_no_indicator_found_raises(self, monthly_publication_delays, indicators):
        """No indicator found: ``ValueError`` listing the request."""
        with pytest.raises(ValueError, match="No data found for specified indicators"):
            calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators=indicators)

    def test_indicators_given_as_a_string_raise(self, monthly_publication_delays):
        """A single name instead of a list is refused (pandas ``TypeError``)."""
        with pytest.raises(TypeError, match="list-like"):
            calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators='PIB')

    def test_none_selects_every_indicator(self, monthly_publication_delays):
        """``indicators=None`` keeps all the indicators."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', indicators=None)

        assert list(result.index) == ['PIB', 'inflation']

    def test_aggregation_across_entities_by_default(self, monthly_publication_delays):
        """Across both countries: PIB 131, 101, 70 (France) and 101, 70 (Germany), start reference, median 101.

        Valeurs d'or : tri 70, 70, 101, 101, 131 -> médiane 101 ; inflation : 101, 70, 70 -> médiane 70.
        """
        result = calculate_applicable_delay(monthly_publication_delays, 'start', 'M')

        assert result['delay'].to_dict() == {'PIB': 101, 'inflation': 70}

    def test_aggregation_per_entity(self, monthly_publication_delays):
        """Per (country, indicator) couple, mean of the start-reference delays.

        Valeurs d'or : (FR, PIB) (131 + 101 + 70) / 3 = 100,67 ; (FR, infl.) (101 + 70) / 2 = 85,5 ;
        (DE, PIB) 85,5 ; (DE, infl.) 70.
        """
        result = calculate_applicable_delay(monthly_publication_delays, 'start', 'M', aggregate_by_panel=True,
                                            aggregation_method='mean')

        assert result['delay'].tolist() == pytest.approx([302 / 3, 85.5, 85.5, 70])

    def test_three_level_panel(self):
        """A three-level index keeps its names with ``aggregate_by_panel`` and aggregates by indicator otherwise."""
        keys = [('EU', 'France', 'PIB'), ('EU', 'Germany', 'PIB'), ('US', 'Texas', 'PIB')]
        delays = _frame([_monthly(delay=14), _monthly(delay=16), _monthly(delay=18)], keys,
                        names=('region', 'country', 'indicator'))

        by_panel = calculate_applicable_delay(delays, 'end', 'M', aggregate_by_panel=True)
        by_indicator = calculate_applicable_delay(delays, 'end', 'M')

        assert (list(by_panel.index.names), by_panel['delay'].tolist(), by_indicator['delay'].tolist()) == (
            ['region', 'country', 'indicator'], [14, 16, 18], [16])

    def test_non_standard_index_names(self):
        """The names of the levels are free: only the position (the last level is the indicator) matters."""
        delays = _frame([_monthly(delay=14), _monthly(delay=16)], [('FR', 'GDP'), ('DE', 'GDP')],
                        names=('pays (code)', 'indicateur é'))

        result = calculate_applicable_delay(delays, 'end', 'M', aggregate_by_panel=True)

        assert list(result.index.names) == ['pays (code)', 'indicateur é']

    def test_indicator_names_with_spaces_and_accents(self):
        """Indicator names are not interpreted."""
        delays = _frame([_monthly(delay=14), _monthly(delay=16)], ['taux de chômage', 'PIB (T/T-1)'])

        assert list(calculate_applicable_delay(delays, 'end', 'M').index) == ['PIB (T/T-1)', 'taux de chômage']

    def test_unsorted_rows_give_the_same_result(self, monthly_publication_delays):
        """Shuffled rows give the same (sorted) result."""
        shuffled = monthly_publication_delays.sample(frac=1.0, random_state=3)

        pd.testing.assert_frame_equal(
            calculate_applicable_delay(shuffled, 'end', 'Q', aggregate_by_panel=True),
            calculate_applicable_delay(monthly_publication_delays, 'end', 'Q', aggregate_by_panel=True),
        )

    def test_duplicated_index_keys_are_distinct_observations(self):
        """Rows sharing an index key are separate observations of the same group."""
        delays = _frame([_monthly('2023-10', 10), _monthly('2023-11', 12), _monthly('2023-12', 14)])

        result = calculate_applicable_delay(delays, 'end', 'M', aggregation_method='mean')

        assert (result['delay'].tolist(), result['n_observations'].tolist()) == ([12.0], [3])

    @pytest.mark.parametrize(
        ('keys', 'names'),
        [(['PIB', 'PIB'], (None,)), ([('FR', 'PIB'), ('DE', 'PIB')], (None, None))],
        ids=['unnamed-index', 'unnamed-multiindex'],
    )
    def test_unnamed_index_levels(self, keys, names):
        """An index without level names is read by position: the last level is the indicator (ANO-DELAYS-014)."""
        delays = _frame([_monthly(delay=14), _monthly(delay=16)], keys, names=names)

        assert _delay(delays, 'end', 'M') == [15]

    def test_unnamed_index_levels_stay_unnamed_in_the_panel_result(self):
        """The panel aggregation of an unnamed ``MultiIndex`` keeps it unnamed, one row per couple."""
        delays = _frame([_monthly(delay=14), _monthly(delay=16)], [('FR', 'PIB'), ('DE', 'PIB')], names=(None, None))

        result = calculate_applicable_delay(delays, 'end', 'M', aggregate_by_panel=True)

        assert (list(result.index.names), result['delay'].tolist()) == ([None, None], [16, 14])

    def test_duplicated_level_names_are_read_by_position(self):
        """Two levels sharing a name are not an obstacle either: only positions matter."""
        delays = _frame([_monthly(delay=14), _monthly(delay=16)], [('FR', 'PIB'), ('DE', 'PIB')], names=('x', 'x'))

        assert _delay(delays, 'end', 'M', aggregate_by_panel=True) == [16, 14]


# =============================================================================
# Valeurs extrêmes, dégénérées et fuseaux
# =============================================================================
class TestDelayMagnitudes:
    """Zero, negative, huge and degenerate delays."""

    @pytest.mark.parametrize(('target', 'expected'), [('end', 0), ('start', 31)], ids=['end', 'start'])
    def test_zero_delay(self, target, expected):
        """Published exactly at the period end: 0 d from the end, 31 d (December) from the start."""
        assert _delay(_frame([_monthly('2023-12', 0)]), target, 'M') == [expected]

    @pytest.mark.parametrize(('target', 'expected'), [('end', -7), ('start', 24)], ids=['end', 'start'])
    def test_negative_delay(self, target, expected):
        """Published 7 d before the period end (December 25th): -7 d from the end, kept negative; 24 d from the start."""
        assert _delay(_frame([_monthly('2023-12', -7)]), target, 'M') == [expected]

    def test_negative_delay_is_kept_for_a_quarterly_target(self):
        """A delay negative from the month end stays so from the quarter end (December 25th, Q4 ends January 1st): -7 d."""
        assert _delay(_frame([_monthly('2023-12', -7)]), 'end', 'Q') == [-7]

    @pytest.mark.parametrize(('delay', 'target', 'expected'), [(1440, 'end', 1440), (1440, 'start', 1471), (36_500, 'end', 36_500)],
                             ids=['four-years-end', 'four-years-start', 'a-century'])
    def test_very_large_delay(self, delay, target, expected):
        """Very large delays are kept; counted from the start of January 2020, 31 d are added."""
        delays = _frame([_row('2020-01-15', '2020-01-01', '2020-02-01', delay)])

        assert _delay(delays, target, 'M') == [expected]

    def test_very_large_delay_in_seconds(self):
        """A century in seconds is exact (3.15e9 s)."""
        delays = _frame([_monthly('2023-12', 36_500 * 86_400, unit='second')])

        assert _delay(delays, 'end', 'M') == [36_500 * 86_400]

    def test_fractional_delay_is_rounded_up(self):
        """Converted delays are whole numbers: 1.5 s becomes 2 s (ceiling)."""
        assert _delay(_frame([_monthly('2023-12', 1.5, unit='second')]), 'end', 'M') == [2]

    def test_single_observation(self):
        """One observation: its own delay, counted once."""
        result = calculate_applicable_delay(_frame([_monthly('2023-12', 14)]), 'end', 'M')

        assert (result['delay'].tolist(), result['n_observations'].tolist()) == ([14], [1])

    def test_time_zone_aware_dates(self):
        """Time-zone-aware dates (UTC) give the same delays as naive ones."""
        delays = _frame([_monthly('2023-12', 14)])
        for column in ['observation_date', 'period_start', 'period_end', 'download_date']:
            delays[column] = delays[column].dt.tz_localize('UTC')

        assert _delay(delays, 'start', 'M') == [45]

    @pytest.mark.parametrize('resolution', ['s', 'ms', 'us', 'ns'])
    def test_datetime_resolution_does_not_matter(self, resolution):
        """``datetime64`` columns of any resolution give the same delay."""
        delays = _frame([_monthly('2023-12', 14)])
        for column in ['observation_date', 'period_start', 'period_end', 'download_date']:
            delays[column] = delays[column].astype(f'datetime64[{resolution}]')

        assert _delay(delays, 'start', 'M') == [45]

    def test_empty_frame_raises_a_clear_error(self):
        """A frame without any row (valid columns) is refused, saying so (ANO-DELAYS-013)."""
        with pytest.raises(ValueError, match="publication_delays has no row"):
            calculate_applicable_delay(_frame([]), 'end', 'M')

    @pytest.mark.parametrize('rows', ['nan-delay', 'no-frequency'])
    def test_unknown_delay_row_is_kept_with_a_nan_delay(self, rows):
        """A row whose delay is ``NaN`` or whose frequency is ``None`` gives a ``NaN`` delay, kept in the panel result.

        ANO-DELAYS-011: the unknown row used to make the whole call fail. Here (France, PIB) is valid (14 d),
        (Germany, PIB) is unknown.
        """
        unknown = _monthly('2023-12', np.nan) if rows == 'nan-delay' else _monthly('2023-12', 14, frequency=None)
        delays = _frame([_monthly('2023-12', 14), unknown], [('France', 'PIB'), ('Germany', 'PIB')],
                        names=('country', 'indicator'))

        result = calculate_applicable_delay(delays, 'end', 'M', aggregate_by_panel=True)

        assert (result['delay'].tolist()[0], np.isnan(result['delay'].tolist()[1]), result['n_observations'].tolist()) == (
            14, True, [1, 0])

    @pytest.mark.parametrize('rows', ['nan-delay', 'no-frequency'])
    def test_unknown_delay_row_is_ignored_by_the_aggregation(self, rows):
        """Aggregated by indicator, the unknown row is left out of the aggregation and of ``n_observations``."""
        unknown = _monthly('2023-12', np.nan) if rows == 'nan-delay' else _monthly('2023-12', 14, frequency=None)
        delays = _frame([_monthly('2023-11', 14), unknown])

        result = calculate_applicable_delay(delays, 'end', 'M')

        assert (result['delay'].tolist(), result['n_observations'].tolist()) == ([14], [1])

    @pytest.mark.parametrize('method', ['median', 'sum', 'max', 'mean'])
    def test_group_of_unknown_delays_has_a_nan_delay(self, method):
        """A group made only of unknown delays gets ``NaN`` (a ``sum`` of nothing is not 0) and zero observations."""
        delays = _frame([_monthly('2023-11', np.nan), _monthly('2023-12', 14, frequency=None)])

        result = calculate_applicable_delay(delays, 'end', 'M', aggregation_method=method)

        assert (np.isnan(result['delay'].iloc[0]), result['n_observations'].tolist()) == (True, [0])

    def test_unknown_delay_survives_the_unit_conversion_and_the_reference_change(self):
        """A ``NaN`` delay stays ``NaN`` through ``unit`` and a start reference, the known one being converted."""
        delays = _frame([_monthly('2023-11', 14), _monthly('2023-12', np.nan)], [('FR', 'PIB'), ('DE', 'PIB')],
                        names=('country', 'indicator'))

        result = calculate_applicable_delay(delays, 'start', 'M', unit='h', aggregate_by_panel=True)

        # Valeur d'or : novembre publié 14 j après sa fin (1er déc.) -> 44 j depuis le 1er nov. = 1056 h
        assert (result['delay'].tolist()[1], np.isnan(result['delay'].tolist()[0]), result['unit'].unique().tolist()) == (
            1056, True, ['h'])

    def test_unknown_source_frequency_is_not_validated_against_the_target(self):
        """The unknown source frequency of a ``NaN`` row does not hide an unsupported target frequency."""
        delays = _frame([_monthly('2023-12', 14, frequency=None)])

        with pytest.raises(ValueError, match="Unsupported frequency: zz"):
            calculate_applicable_delay(delays, 'end', 'zz')


# =============================================================================
# Contrat avec compare_and_detect_delays
# =============================================================================
class TestContractWithDataManager:
    """The output of ``compare_and_detect_delays`` is consumed as is; gold values by calendar."""

    def test_fixture_holds_the_hand_computed_delays(self, monthly_publication_delays):
        """Guard: the detected delays are the ones of the fixture docstring (131, 101, 70 d ...)."""
        detected = monthly_publication_delays['delay'].groupby(level=['country', 'column']).apply(list).to_dict()

        assert detected == {('France', 'PIB'): [131.0, 101.0, 70.0], ('France', 'inflation'): [101.0, 70.0],
                            ('Germany', 'PIB'): [101.0, 70.0], ('Germany', 'inflation'): [70.0]}

    def test_data_manager_output_is_accepted_without_adaptation(self, monthly_publication_delays):
        """The frame returned by ``compare_and_detect_delays`` is a valid input, as is."""
        assert set(REQUIRED_COLUMNS) <= set(monthly_publication_delays.columns)

    def test_monthly_to_monthly_from_the_end(self, monthly_publication_delays):
        """Counted from the month ends (May 1st, June 1st, July 1st): 101, 70 and 40 d, maximum per couple."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregate_by_panel=True,
                                            aggregation_method='max')

        # Valeurs d'or : 10 août - 1er mai = 101 j, - 1er juin = 70 j, - 1er juillet = 40 j ; maximum par couple :
        # (FR, PIB) avril, mai, juin -> 101 ; (FR, inflation) mai, juin -> 70 ; (DE, PIB) mai, juin -> 70 ;
        # (DE, inflation) juin -> 40
        assert result['delay'].tolist() == [101, 70, 70, 40]

    def test_monthly_to_quarterly_from_the_start(self, monthly_publication_delays):
        """April, May and June are all in Q2: counted from April 1st, 131 d for every observation."""
        result = calculate_applicable_delay(monthly_publication_delays, 'start', 'Q', aggregate_by_panel=True)

        assert result['delay'].tolist() == [131, 131, 131, 131]

    def test_monthly_to_quarterly_from_the_end(self, monthly_publication_delays):
        """Counted from the Q2 end (July 1st): 31 + 9 = 40 d to August 10th, for every observation."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'Q')

        assert result['delay'].tolist() == [40, 40]

    def test_counts_of_the_detected_observations(self, monthly_publication_delays):
        """Observations per couple: 3, 2, 2 and 1 (the rows detected by ``compare_and_detect_delays``)."""
        result = calculate_applicable_delay(monthly_publication_delays, 'end', 'M', aggregate_by_panel=True)

        assert result['n_observations'].tolist() == [3, 2, 2, 1]

    def test_quarterly_delays_dated_at_period_start(self, quarterly_publication_delays):
        """Quarter-start dates map onto the first month: Q1 2023 -> January, Q2 2023 -> April; from the start unchanged."""
        result = calculate_applicable_delay(quarterly_publication_delays, 'start', 'M', aggregate_by_panel=True,
                                            aggregation_method='max')

        # Valeurs d'or : 221 j (1er janv.) et 131 j (1er avril) pour la France, 131 j pour l'Allemagne
        assert result['delay'].tolist() == [221, 131]

    def test_quarterly_to_monthly_from_the_end(self, quarterly_publication_delays):
        """From the end of the first month: Feb 1st (190 d) and May 1st (101 d)."""
        result = calculate_applicable_delay(quarterly_publication_delays, 'end', 'M', aggregate_by_panel=True,
                                            aggregation_method='mean')

        # France : (190 + 101) / 2 = 145,5 ; Allemagne : 101
        assert result['delay'].tolist() == [145.5, 101]

    def test_quarterly_to_quarterly_from_the_end(self, quarterly_publication_delays):
        """Quarter ends are April 1st (131 d) and July 1st (40 d)."""
        result = calculate_applicable_delay(quarterly_publication_delays, 'end', 'Q', aggregate_by_panel=True,
                                            aggregation_method='max')

        assert result['delay'].tolist() == [131, 40]

    def test_time_series_end_reference(self, simple_time_series_delays):
        """First download of a series: December observations give 14 d from the end, November 45 d."""
        result = calculate_applicable_delay(simple_time_series_delays, 'end', 'M')

        assert result['delay'].to_dict() == {'PIB': 14, 'inflation': 14, 'unemployment': 45}

    def test_time_series_start_reference(self, simple_time_series_delays):
        """From the start: 45 d for December (31 + 14), 75 d for November (30 + 31 + 14)."""
        result = calculate_applicable_delay(simple_time_series_delays, 'start', 'M')

        assert result['delay'].to_dict() == {'PIB': 45, 'inflation': 45, 'unemployment': 75}

    def test_time_series_to_quarterly(self, simple_time_series_delays):
        """November and December belong to Q4 (October 1st = 106 d before the 15th of January)."""
        result = calculate_applicable_delay(simple_time_series_delays, 'start', 'Q')

        assert result['delay'].to_dict() == {'PIB': 106, 'inflation': 106, 'unemployment': 106}

    def test_seconds_from_the_data_manager(self, delays_in_seconds):
        """Delays detected in seconds keep their unit through the calculator."""
        result = calculate_applicable_delay(delays_in_seconds, 'end', 'M')

        assert result['unit'].tolist() == ['second']

    @pytest.mark.parametrize(
        ('first_label', 'rule', 'periods', 'frequency', 'download', 'start_delay', 'end_delay'),
        [
            ('2023-01-31', 'ME', 4, 'M', '2023-06-15', 75, 45),
            ('2022-03-31', 'QE', 6, 'Q', '2023-08-10', 131, 40),
            ('2020-12-31', 'YE', 4, 'Y', '2024-03-15', 439, 74),
        ],
        ids=['month-end', 'quarter-end', 'year-end'],
    )
    def test_period_end_labelled_dates(self, first_label, rule, periods, frequency, download, start_delay, end_delay):
        """Data labelled on the last day of the period (``ME`` / ``QE`` / ``YE``), same frequency as the target.

        Valeurs d'or : avril 2023 (dernière date 2023-04-30) téléchargé le 15 juin = 75 j / 45 j ; T2 2023 (2023-06-30)
        téléchargé le 10 août = 131 j / 40 j ; année 2023 (2023-12-31) téléchargée le 15 mars 2024 = 439 j / 74 j.
        """
        dates = pd.date_range(first_label, periods=periods, freq=rule)
        data = pd.DataFrame({'PIB': np.arange(float(periods))}, index=dates)
        delays = _detect(data, None, download_date=download, reference_point='start')

        assert (_delay(delays, 'start', frequency), _delay(delays, 'end', frequency)) == ([start_delay], [end_delay])

    @pytest.mark.parametrize(
        ('target', 'start_delay', 'end_delay'),
        [('M', 105, 74), ('Q', 166, 74)],
        ids=['year-end-to-month', 'year-end-to-quarter'],
    )
    def test_year_end_labelled_dates_to_a_higher_frequency(self, target, start_delay, end_delay):
        """A year labelled December 31st maps onto the last month (December) or the last quarter (Q4).

        Valeurs d'or, année 2023 téléchargée le 15 mars 2024 : décembre -> début le 1er déc. = 31 + 74 = 105 j, fin le
        1er janv. : 74 j ; T4 -> début le 1er oct. = 92 + 74 = 166 j.
        """
        data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0]}, index=pd.date_range('2020-12-31', periods=4, freq='YE'))
        delays = _detect(data, None, download_date='2024-03-15', reference_point='start')

        assert (_delay(delays, 'start', target), _delay(delays, 'end', target)) == ([start_delay], [end_delay])

    def test_all_changes_detection_mode_gives_a_valid_input(self):
        """Delays detected with ``all_changes`` (revisions included) are also accepted."""
        months = pd.date_range('2023-01-01', periods=4, freq='MS')
        existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0]}, index=months)
        new = pd.DataFrame({'PIB': [1.0, 2.5, 3.0, 4.0]}, index=months)
        detected = _detect(new, existing, download_date='2023-06-15', detection_mode='all_changes')

        # Valeur d'or : révision de février, téléchargée le 15 juin : 1er févr. -> 15 juin = 28 + 31 + 30 + 31 + 14 = 134 j
        assert _delay(detected, 'start', 'M') == [134]

    @pytest.fixture
    def panel_with_an_undetectable_couple(self) -> pd.DataFrame:
        """Delays of a panel where entity ``B`` is observed once (frequency undetectable, delay ``NaN``).

        Valeur d'or : (A, PIB) dernière observation le 1er avril 2023, téléchargée le 15 juin : 30 + 31 + 14 = 75 j.
        """
        index = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2023-01-01', periods=4, freq='MS')],
                                           names=['country', 'date'])
        panel = pd.DataFrame({'PIB': [1.0, 2, 3, 4, 1, np.nan, np.nan, np.nan]}, index=index)

        return _detect(panel, None, download_date='2023-06-15')

    def test_undetectable_frequency_does_not_break_the_chain(self, panel_with_an_undetectable_couple):
        """One entity observed once (frequency undetectable) does not prevent the delay of the others (ANO-DELAYS-011)."""
        result = calculate_applicable_delay(panel_with_an_undetectable_couple, 'start', 'M')

        assert (result['delay'].tolist(), result['n_observations'].tolist()) == ([75], [1])

    def test_undetectable_couple_is_kept_with_a_nan_delay(self, panel_with_an_undetectable_couple):
        """Per couple, the undetectable one stays in the result with a ``NaN`` delay and no observation."""
        result = calculate_applicable_delay(panel_with_an_undetectable_couple, 'start', 'M', aggregate_by_panel=True)

        assert (list(result.index), result['delay'].isna().tolist(), result['n_observations'].tolist()) == (
            [('A', 'PIB'), ('B', 'PIB')], [False, True], [1, 0])
