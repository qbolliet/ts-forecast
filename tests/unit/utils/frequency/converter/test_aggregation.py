"""Unit tests of ``FrequencyConverter.aggregate_to_lower_frequency``.

Scope: the nine numeric aggregation methods (``mean``, ``sum``, ``first``,
``last``, ``min``, ``max``, ``median``, ``std``, ``count``) on golden values,
the chaining M → Q → Y, the start / end labels of the target periods, NaN
handling (including periods without any observation), ``full_periods_only`` on
DataFrames and when the source frequency cannot be established, the boolean
methods on an irregular index, unsorted and empty inputs, invalid arguments,
and the additivity property (sum of a calendar disaggregation = identity),
which ties ``aggregate_to_lower_frequency`` to ``count_subperiods_per_period``.

The calendar count of ``full_periods_only`` on Series and the boolean methods on
regular grids are covered by ``test_subperiods.py`` and
``test_boolean_aggregation.py``.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter() -> FrequencyConverter:
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


@pytest.fixture
def monthly_with_gap() -> pd.Series:
    """Six month ends of 2024, April missing: [1, 4, 2, NaN, 6, 3]."""
    return pd.Series(
        [1.0, 4.0, 2.0, np.nan, 6.0, 3.0],
        index=pd.date_range('2024-01-31', periods=6, freq='ME'),
    )


# =============================================================================
# Méthodes numériques
# =============================================================================

class TestNumericAggregationMethods:
    """Each numeric method reduces the sub-periods of a target period, NaN skipped."""

    # Valeurs d'or : T1 = [1, 4, 2], T2 = [NaN, 6, 3] (NaN ignoré par pandas)
    # - std (ddof=1) : T1 = sqrt(((-4/3)² + (5/3)² + (-1/3)²) / 2) = sqrt(7/3) ;
    #   T2 = sqrt((1.5² + 1.5²) / 1) = sqrt(4.5)
    # - first / last : première / dernière valeur NON manquante de la période
    @pytest.mark.parametrize(
        'method, expected',
        [
            pytest.param('mean', [7 / 3, 4.5], id='mean'),
            pytest.param('sum', [7.0, 9.0], id='sum'),
            pytest.param('first', [1.0, 6.0], id='first-skips-nan'),
            pytest.param('last', [2.0, 3.0], id='last'),
            pytest.param('min', [1.0, 3.0], id='min'),
            pytest.param('max', [4.0, 6.0], id='max'),
            pytest.param('median', [2.0, 4.5], id='median'),
            pytest.param('std', [np.sqrt(7 / 3), np.sqrt(4.5)], id='std'),
            pytest.param('count', [3, 2], id='count-non-nan'),
        ],
    )
    def test_monthly_to_quarterly_golden_values(self, converter, monthly_with_gap, method, expected):
        """Golden values of every numeric method, monthly to quarterly."""
        result = converter.aggregate_to_lower_frequency(monthly_with_gap, 'QE', method=method)
        assert result.tolist() == pytest.approx(expected)

    def test_labels_are_the_target_period_ends(self, converter, monthly_with_gap):
        """A 'QE' target labels each quarter by its last day."""
        result = converter.aggregate_to_lower_frequency(monthly_with_gap, 'QE', method='sum')
        assert list(result.index) == [pd.Timestamp('2024-03-31'), pd.Timestamp('2024-06-30')]

    def test_unsupported_method_raises(self, converter, monthly_with_gap):
        """An unknown method name is rejected with its name in the message."""
        with pytest.raises(ValueError, match='Unsupported aggregation method: foo'):
            converter.aggregate_to_lower_frequency(monthly_with_gap, 'QE', method='foo')

    def test_integer_mean_is_float(self, converter):
        """The mean of an integer series is a float series."""
        series = pd.Series([1, 2, 4], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
        result = converter.aggregate_to_lower_frequency(series, 'QE', method='mean')
        # Valeur d'or : (1 + 2 + 4) / 3
        assert result.dtype == np.float64 and result.iloc[0] == pytest.approx(7 / 3)


class TestAggregationChain:
    """Aggregating M → Q then Q → Y equals aggregating M → Y directly."""

    # Données : mois 1..24 de 2023-2024 ; valeurs d'or annuelles
    # - sum : 1+…+12 = 78, 13+…+24 = 222 ; mean : 78/12 = 6.5, 222/12 = 18.5
    # - last / max : 12 et 24 (séries croissantes)
    @pytest.mark.parametrize(
        'method, expected',
        [
            pytest.param('sum', [78.0, 222.0], id='sum'),
            pytest.param('mean', [6.5, 18.5], id='mean-of-equal-sized-quarters'),
            pytest.param('last', [12.0, 24.0], id='last'),
            pytest.param('max', [12.0, 24.0], id='max'),
        ],
    )
    def test_chained_equals_direct(self, converter, method, expected):
        """Two chained aggregations give the golden yearly values of the direct one."""
        monthly = pd.Series(
            np.arange(1.0, 25.0), index=pd.date_range('2023-01-31', periods=24, freq='ME')
        )
        quarterly = converter.aggregate_to_lower_frequency(monthly, 'QE', method=method)
        chained = converter.aggregate_to_lower_frequency(quarterly, 'YE', method=method)
        direct = converter.aggregate_to_lower_frequency(monthly, 'YE', method=method)

        assert chained.tolist() == pytest.approx(expected)
        assert direct.tolist() == pytest.approx(expected)


class TestAggregationLabels:
    """The target position (S / E) only changes the labels, never the periods."""

    # Sommes d'or des mois 1..12 de 2024 : trimestres 6, 15, 24, 33 ; année 78
    _QUARTER_SUMS = [6.0, 15.0, 24.0, 33.0]

    @pytest.mark.parametrize('source_freq', ['MS', 'ME'], ids=['source-start', 'source-end'])
    @pytest.mark.parametrize(
        'target_freq, labels, values',
        [
            pytest.param('QS', ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
                         _QUARTER_SUMS, id='QS'),
            pytest.param('QE', ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
                         _QUARTER_SUMS, id='QE'),
            pytest.param('YS', ['2024-01-01'], [78.0], id='YS'),
            pytest.param('YE', ['2024-12-31'], [78.0], id='YE'),
        ],
    )
    def test_target_position_labels(self, converter, source_freq, target_freq, labels, values):
        """Same calendar periods whatever the source position, labelled at the target position."""
        start = '2024-01-01' if source_freq == 'MS' else '2024-01-31'
        monthly = pd.Series(np.arange(1.0, 13.0), index=pd.date_range(start, periods=12, freq=source_freq))

        result = converter.aggregate_to_lower_frequency(monthly, target_freq, method='sum')

        expected = pd.Series(values, index=pd.DatetimeIndex(labels))
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_index_name_is_kept(self, converter):
        """The name of the source index survives the aggregation."""
        monthly = pd.Series(
            [1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME', name='date')
        )
        result = converter.aggregate_to_lower_frequency(monthly, 'QE', method='sum')
        assert result.index.name == 'date'


# =============================================================================
# Valeurs manquantes et périodes vides
# =============================================================================

class TestPeriodsWithoutObservation:
    """A target period whose sub-periods are all missing."""

    @pytest.fixture
    def second_quarter_missing(self) -> pd.Series:
        """Monthly 2024 H1 with the whole second quarter missing."""
        return pd.Series(
            [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
            index=pd.date_range('2024-01-31', periods=6, freq='ME'),
        )

    @pytest.mark.parametrize('method', ['mean', 'median', 'min', 'max', 'first', 'last', 'std'])
    def test_reductions_give_nan(self, converter, second_quarter_missing, method):
        """Every reduction of an empty set gives NaN."""
        result = converter.aggregate_to_lower_frequency(second_quarter_missing, 'QE', method=method)
        assert pd.isna(result.iloc[1])

    def test_count_gives_zero(self, converter, second_quarter_missing):
        """No observation in the period: the count is 0."""
        result = converter.aggregate_to_lower_frequency(second_quarter_missing, 'QE', method='count')
        assert result.tolist() == [3, 0]

    def test_sum_gives_nan(self, converter, second_quarter_missing):
        """No observation in the period: the sum is missing, not a genuine zero."""
        result = converter.aggregate_to_lower_frequency(second_quarter_missing, 'QE', method='sum')
        # Valeur d'or : T1 = 1 + 2 + 3 = 6 ; T2 sans aucune observation → NaN (comme 'mean')
        assert result.iloc[0] == 6.0 and pd.isna(result.iloc[1])


class TestFullPeriodsOnlyOnDataFrame:
    """``full_periods_only`` masks the incomplete periods column by column."""

    @pytest.fixture
    def two_columns(self) -> pd.DataFrame:
        """``a`` misses March, ``b`` misses June (2024 H1, month ends)."""
        return pd.DataFrame(
            {
                'a': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                'b': [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
            },
            index=pd.date_range('2024-01-31', periods=6, freq='ME'),
        )

    # Valeurs d'or : a → T1 incomplet (2 mois sur 3), T2 = 4 + 5 + 6 = 15 ;
    # b → T1 = 10 + 20 + 30 = 60, T2 incomplet ; moyennes : a T2 = 5, b T1 = 20
    @pytest.mark.parametrize(
        'method, expected_a, expected_b',
        [
            pytest.param('sum', [np.nan, 15.0], [60.0, np.nan], id='sum'),
            pytest.param('mean', [np.nan, 5.0], [20.0, np.nan], id='mean'),
        ],
    )
    def test_each_column_is_masked_independently(self, converter, two_columns, method, expected_a, expected_b):
        """A gap in one column never masks the other."""
        result = converter.aggregate_to_lower_frequency(two_columns, 'QE', method=method, full_periods_only=True)
        expected = pd.DataFrame({'a': expected_a, 'b': expected_b}, index=result.index)
        pd.testing.assert_frame_equal(result, expected)

    def test_partial_period_at_the_start_of_the_grid(self, converter):
        """A grid starting in February leaves the first quarter incomplete."""
        # Grille février → juin : T1 ne compte que février et mars (2 mois sur 3)
        series = pd.Series(
            [2.0, 3.0, 4.0, 5.0, 6.0], index=pd.date_range('2024-02-29', periods=5, freq='ME')
        )
        result = converter.aggregate_to_lower_frequency(series, 'QE', method='sum', full_periods_only=True)
        # Valeur d'or : T1 masqué, T2 = 4 + 5 + 6 = 15
        assert result.tolist() == pytest.approx([np.nan, 15.0], nan_ok=True)


class TestCoverageGuardsWithoutSourceFrequency:
    """Without a detectable (or supplied) source frequency, no coverage guard applies."""

    @pytest.fixture
    def irregular(self) -> pd.Series:
        """Four observations at irregular dates: 3 in January, 1 in February."""
        return pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=pd.DatetimeIndex(['2024-01-01', '2024-01-03', '2024-01-10', '2024-02-20']),
        )

    def test_full_periods_only_is_skipped(self, converter, irregular):
        """No expected sub-period count exists: every month is kept as is."""
        result = converter.aggregate_to_lower_frequency(irregular, 'MS', method='sum', full_periods_only=True)
        # Valeurs d'or : janvier = 1 + 2 + 3 = 6, février = 4 (aucun masquage possible)
        assert result.tolist() == [6.0, 4.0]

    def test_all_does_not_require_full_coverage(self, converter, irregular):
        """'all' reduces the present values only: a month with one True value is True."""
        result = converter.aggregate_to_lower_frequency(irregular > 1, 'MS', method='all')
        # Valeurs d'or : janvier contient 1 > 1 == False ; février ne contient que 4 > 1
        assert result.tolist() == [False, True]

    def test_single_observation_skips_the_guard(self, converter):
        """One date, no detectable frequency: the lone observation is aggregated as is."""
        single = pd.Series([5.0], index=pd.DatetimeIndex(['2024-01-31']))
        result = converter.aggregate_to_lower_frequency(single, 'QE', method='sum', full_periods_only=True)
        assert result.tolist() == [5.0]

    def test_supplied_source_frequency_restores_the_guard(self, converter, irregular):
        """``source_freq='D'`` gives an expected count (31, 29 days): both months are masked."""
        result = converter.aggregate_to_lower_frequency(
            irregular, 'MS', method='sum', full_periods_only=True, source_freq='D'
        )
        assert result.isna().all()

    def test_invalid_source_frequency_raises_for_all(self, converter, irregular):
        """``method='all'`` rejects an unknown supplied source frequency."""
        with pytest.raises(ValueError, match='Unsupported frequency: foo'):
            converter.aggregate_to_lower_frequency(irregular > 1, 'MS', method='all', source_freq='foo')

    def test_invalid_source_frequency_raises_for_full_periods_only(self, converter, irregular):
        """``full_periods_only`` rejects an unknown supplied source frequency, like 'all'."""
        with pytest.raises(ValueError):
            converter.aggregate_to_lower_frequency(
                irregular, 'MS', method='sum', full_periods_only=True, source_freq='foo'
            )


# =============================================================================
# Cas limites et arguments invalides
# =============================================================================

class TestAggregationEdgeCases:
    """Unsorted, empty and panel inputs; invalid target frequencies."""

    def test_unsorted_input_gives_the_sorted_result(self, converter):
        """The rows are aggregated by period whatever their order."""
        monthly = pd.Series(
            np.arange(1.0, 7.0), index=pd.date_range('2024-01-31', periods=6, freq='ME')
        )
        result = converter.aggregate_to_lower_frequency(monthly.iloc[::-1], 'QE', method='sum')
        # Valeurs d'or : 1 + 2 + 3 = 6, 4 + 5 + 6 = 15, dans l'ordre chronologique
        assert result.tolist() == [6.0, 15.0]

    def test_unsorted_input_keeps_the_coverage_guard(self, converter, monthly_with_gap):
        """Reversed rows are sorted before the source frequency is detected."""
        result = converter.aggregate_to_lower_frequency(
            monthly_with_gap.iloc[::-1], 'QE', method='sum', full_periods_only=True
        )
        # Valeurs d'or : T1 complet (1 + 4 + 2 = 7) ; T2 privé d'avril, masqué
        assert result.tolist() == pytest.approx([7.0, np.nan], nan_ok=True)

    def test_empty_series_raises(self, converter):
        """No row, nothing to aggregate: the input is refused."""
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match='Cannot aggregate empty data'):
            converter.aggregate_to_lower_frequency(empty, 'QE', method='sum')

    def test_invalid_target_frequency_raises(self, converter, monthly_with_gap):
        """An anchor unknown to pandas is reported with the target frequency."""
        with pytest.raises(ValueError, match="Invalid target frequency 'QS-FOO'"):
            converter.aggregate_to_lower_frequency(monthly_with_gap, 'QS-FOO', method='sum')

    def test_user_label_is_not_an_offset(self, converter, monthly_with_gap):
        """The target is a pandas offset: user labels ('monthly') are rejected (``013d929``)."""
        with pytest.raises(ValueError, match="Invalid target frequency 'quarterly'"):
            converter.aggregate_to_lower_frequency(monthly_with_gap, 'quarterly', method='sum')

    def test_panel_must_go_through_convert_frequency(self, converter):
        """A MultiIndex is not resampled directly: panels go through ``convert_frequency``."""
        index = pd.MultiIndex.from_product(
            [['A', 'B'], pd.date_range('2024-01-31', periods=3, freq='ME')], names=['entity', 'date']
        )
        panel = pd.DataFrame({'x': np.arange(6.0)}, index=index)
        with pytest.raises(TypeError):
            converter.aggregate_to_lower_frequency(panel, 'QE', method='sum')


# =============================================================================
# Propriété d'additivité
# =============================================================================

# Couples (fréquence basse, alias Period de la base basse, fréquence haute), sur
# deux années dont une bissextile : le décompte calendaire doit retrouver 28 ou
# 29 jours en février, 365 ou 366 jours par an
_ADDITIVITY_PAIRS = [
    pytest.param('ME', 'M', 'D', '2023-01-01', '2024-12-31', id='month-end-to-day'),
    pytest.param('MS', 'M', 'D', '2023-01-01', '2024-12-31', id='month-start-to-day'),
    pytest.param('QE', 'Q', 'ME', '2023-01-01', '2024-12-31', id='quarter-end-to-month'),
    pytest.param('QS', 'Q', 'MS', '2023-01-01', '2024-12-31', id='quarter-start-to-month'),
    pytest.param('YS', 'Y', 'MS', '2023-01-01', '2024-12-31', id='year-start-to-month'),
    pytest.param('YE', 'Y', 'D', '2023-01-01', '2024-12-31', id='year-end-to-day-leap'),
]


def _calendar_disaggregation(
    converter: FrequencyConverter,
    low: pd.Series,
    low_freq: str,
    low_period: str,
    high_index: pd.DatetimeIndex,
    high_freq: str,
) -> pd.Series:
    """Spread each low-frequency value evenly over its own sub-periods.

    Args:
        converter: Converter providing the calendar sub-period counts.
        low: Low-frequency values, one per period.
        low_freq: Frequency of ``low`` (with position).
        low_period: Period alias of the low base ('M', 'Q', 'Y').
        high_index: High-frequency grid covering exactly the periods of ``low``.
        high_freq: Frequency of ``high_index``.

    Returns:
        High-frequency series whose sub-periods each carry ``value / count``.
    """
    # Nombre de sous-périodes de chaque période basse (décompte calendaire)
    counts = converter.count_subperiods_per_period(low.index, low_freq, high_freq)
    share = pd.Series(low.to_numpy() / counts, index=low.index.to_period(low_period))

    # Chaque sous-période reçoit la part de la période basse qui la contient
    return pd.Series(share.reindex(high_index.to_period(low_period)).to_numpy(), index=high_index)


class TestSumAdditivity:
    """Summing back an even calendar disaggregation gives the original series.

    The disaggregation spreads each low-frequency value over its sub-periods
    using ``count_subperiods_per_period``; aggregating it by ``'sum'`` with
    ``full_periods_only=True`` must return every value unchanged and mask no
    period — the calendar count (B26 defect, spec §9.3) is exact on both sides.
    """

    @pytest.mark.parametrize('low_freq, low_period, high_freq, start, end', _ADDITIVITY_PAIRS)
    @pytest.mark.parametrize('seed', [0, 1])
    def test_sum_of_disaggregation_is_identity(self, converter, low_freq, low_period, high_freq, start, end, seed):
        """``aggregate(sum, full_periods_only) ∘ disaggregate`` is the identity."""
        # Série basse fréquence aléatoire, reproductible
        low_index = pd.date_range(start, end, freq=low_freq)
        rng = np.random.default_rng(seed)
        low = pd.Series(rng.uniform(10.0, 100.0, len(low_index)), index=low_index)

        # Désagrégation calendaire sur la grille haute fréquence couvrant les mêmes périodes
        high_index = pd.date_range(start, end, freq=high_freq)
        high = _calendar_disaggregation(converter, low, low_freq, low_period, high_index, high_freq)

        result = converter.aggregate_to_lower_frequency(high, low_freq, method='sum', full_periods_only=True)

        pd.testing.assert_series_equal(result, low, check_freq=False, rtol=1e-12)

    @pytest.mark.parametrize('low_freq, low_period, high_freq, start, end', _ADDITIVITY_PAIRS)
    def test_calendar_count_matches_observed_count(self, converter, low_freq, low_period, high_freq, start, end):
        """The expected sub-period count equals the number of dates of each period on the grid."""
        low_index = pd.date_range(start, end, freq=low_freq)
        ones = pd.Series(1.0, index=pd.date_range(start, end, freq=high_freq))

        observed = converter.aggregate_to_lower_frequency(ones, low_freq, method='count')
        expected = converter.count_subperiods_per_period(low_index, low_freq, high_freq)

        np.testing.assert_array_equal(observed.to_numpy(dtype=float), expected)
