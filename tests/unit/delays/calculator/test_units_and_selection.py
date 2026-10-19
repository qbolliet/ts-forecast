"""Units, indicator selection and compatibility with the delay detection.

Scenarios: ceiling rounding to the target ``unit``, indicator and aggregation-level
selection, and the consumption of ``compare_and_detect_delays`` output as is.
"""

import numpy as np
import pandas as pd
import pytest

# Fonction à tester
from tsforecast.delays.calculator import calculate_applicable_delay

# Outils de test partagés du paquet
from tests.unit.delays.calculator.helpers import (
    REQUIRED_COLUMNS,
    TS,
    _delay,
    _detect,
    _frame,
    _monthly,
)


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
def delays_in_seconds() -> pd.DataFrame:
    """One monthly ``PIB`` delay in seconds: December 2023, downloaded 2024-01-15 12:30:45, from the period end.

    Gold: 14 d + 12 h + 30 min + 45 s = 1 209 600 + 43 200 + 1 800 + 45 = 1 254 645 s.
    """
    data = pd.DataFrame({'PIB': np.arange(12.0)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))

    return _detect(data, None, download_date='2024-01-15 12:30:45', reference_point='end', delay_unit='s')


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
        """Without ``unit``, rows of one indicator in days and in seconds are refused, naming the group."""
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

        The conversion works on integer nanoseconds: float seconds would lose 1 µs.
        """
        delays = _frame([_monthly('2023-12', delay, unit='microsecond')])

        assert _delay(delays, target, 'M') == [expected]


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
        """An index without level names is read by position: the last level is the indicator."""
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
        """One entity observed once (frequency undetectable) does not prevent the delay of the others."""
        result = calculate_applicable_delay(panel_with_an_undetectable_couple, 'start', 'M')

        assert (result['delay'].tolist(), result['n_observations'].tolist()) == ([75], [1])

    def test_undetectable_couple_is_kept_with_a_nan_delay(self, panel_with_an_undetectable_couple):
        """Per couple, the undetectable one stays in the result with a ``NaN`` delay and no observation."""
        result = calculate_applicable_delay(panel_with_an_undetectable_couple, 'start', 'M', aggregate_by_panel=True)

        assert (list(result.index), result['delay'].isna().tolist(), result['n_observations'].tolist()) == (
            [('A', 'PIB'), ('B', 'PIB')], [False, True], [1, 0])
