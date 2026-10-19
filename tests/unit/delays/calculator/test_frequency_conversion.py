"""Frequency conversion of publication delays.

Scenarios: hand-computed delays for another target frequency, aggregation by indicator
or by (entity, indicator) and one frequency per indicator given as a dictionary.
"""

import pandas as pd
import pytest

# Fonction à tester
from tsforecast.delays.calculator import calculate_applicable_delay

# Outils de test partagés du paquet
from tests.unit.delays.calculator.helpers import _delay, _frame, _monthly, _row


def _quarterly(observation: str = '2024-03-15', delay=44, **kwargs) -> dict:
    """Build the row of a Q1 2024 observation: period ``[2024-01-01, 2024-04-01)``, download on 2024-05-15 by default."""
    return _row(observation, '2024-01-01', '2024-04-01', delay, frequency='quarterly', **kwargs)


def _annual(observation: str = '2023-06-15', delay=74, **kwargs) -> dict:
    """Build the row of a 2023 observation: period ``[2023-01-01, 2024-01-01)``, download on 2024-03-15 by default."""
    return _row(observation, '2023-01-01', '2024-01-01', delay, frequency='annual', **kwargs)


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
        """A name that is not a pandas aggregation is a ``ValueError`` naming ``aggregation_method``."""
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
        """An indicator missing from the dictionary is named in the error message."""
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
