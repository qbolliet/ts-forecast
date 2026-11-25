"""Output and input contract of ``publication_delays``.

Scenarios: shape, order and types of the returned frame, required columns, reference
point of a delay and zero, negative, huge or degenerate delays.
"""

# Modules de base
import doctest

import numpy as np
import pandas as pd
import pytest

# Fonction à tester
from tsforecast.delays import calculator
from tsforecast.delays.calculator import calculate_applicable_delay

# Outils de test partagés du paquet
from tests.unit.delays.calculator.helpers import (
    REQUIRED_COLUMNS,
    _delay,
    _frame,
    _monthly,
    _row,
)


# Colonnes de sortie, dans l'ordre
OUTPUT_COLUMNS = ['delay', 'unit', 'frequency', 'reference_point', 'n_observations', 'aggregation_method']


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
        """A frame without any row (valid columns) is refused, saying so."""
        with pytest.raises(ValueError, match="publication_delays has no row"):
            calculate_applicable_delay(_frame([]), 'end', 'M')

    @pytest.mark.parametrize('rows', ['nan-delay', 'no-frequency'])
    def test_unknown_delay_row_is_kept_with_a_nan_delay(self, rows):
        """A row whose delay is ``NaN`` or whose frequency is ``None`` gives a ``NaN`` delay, kept in the panel result.

        An unknown row does not make the whole call fail. Here (France, PIB) is valid (14 d),
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
