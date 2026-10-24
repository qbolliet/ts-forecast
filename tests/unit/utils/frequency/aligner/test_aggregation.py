"""Aggregation route of ``FrequencyAligner``.

Scenarios: golden values, aggregation methods, incomplete periods and the index, dtype
and untouched-column contract.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Perturbations partagées
from tests.support.perturbations import shuffle_rows, with_duplicated_rows

# Outils de test partagés du paquet
from tests.unit.frequency.frequency_aligner.helpers import _frame, _panel, _series


# ------------------------------------------------------------------ #
#  Agrégation : valeurs d'or                                          #
# ------------------------------------------------------------------ #

# Cas d'agrégation : (grille source, valeurs, cible, valeurs d'or attendues)
_AGGREGATION_CASES = [
    pytest.param(
        ('2023-01-31', 'ME'), np.arange(1, 13), 'QE',
        # Valeur d'or : 1+2+3, 4+5+6, 7+8+9, 10+11+12 aux fins de trimestre
        {'2023-03-31': 6, '2023-06-30': 15, '2023-09-30': 24, '2023-12-31': 33},
        id='M-end->Q-end',
    ),
    pytest.param(
        ('2023-01-01', 'MS'), np.arange(1, 13), 'Q',
        # Cible sans position : labels ancrés en début de trimestre, comme la source
        {'2023-01-01': 6, '2023-04-01': 15, '2023-07-01': 24, '2023-10-01': 33},
        id='M-start->Q-positionless',
    ),
    pytest.param(
        ('2023-01-01', 'MS'), np.arange(1, 13), 'QS',
        # Cible explicite dans la position de la source : labels aux débuts de trimestre
        {'2023-01-01': 6, '2023-04-01': 15, '2023-07-01': 24, '2023-10-01': 33},
        id='M-start->Q-start',
    ),
    pytest.param(
        ('2023-01-31', 'ME'), np.arange(1, 13), 'Q',
        # Cible sans position sur une source en fin : labels aux fins de trimestre
        {'2023-03-31': 6, '2023-06-30': 15, '2023-09-30': 24, '2023-12-31': 33},
        id='M-end->Q-positionless',
    ),
    pytest.param(
        ('2022-01-01', 'MS'), np.arange(1, 25), 'Y',
        # Valeur d'or : somme(1..12) = 78, somme(13..24) = 222
        {'2022-01-01': 78, '2023-01-01': 222},
        id='M-start->Y',
    ),
    pytest.param(
        ('2022-01-31', 'ME'), np.arange(1, 25), 'YE',
        {'2022-12-31': 78, '2023-12-31': 222},
        id='M-end->Y-end',
    ),
    pytest.param(
        ('2022-01-01', 'QS'), np.arange(1, 9), 'Y',
        # Valeur d'or : 1+2+3+4 = 10, 5+6+7+8 = 26
        {'2022-01-01': 10, '2023-01-01': 26},
        id='Q-start->Y',
    ),
    pytest.param(
        ('2022-03-31', 'QE'), np.arange(1, 9), 'YE',
        {'2022-12-31': 10, '2023-12-31': 26},
        id='Q-end->Y-end',
    ),
]


class TestAggregationGoldenValues:
    """Downsampling sums the sub-periods onto the period labels of the source index."""

    @pytest.mark.parametrize('grid, values, target, expected', _AGGREGATION_CASES)
    def test_sums_land_on_period_labels(self, aligner, grid, values, target, expected):
        """Each complete period carries the sum of its sub-periods, every other row is NaN."""
        df = _frame(values, *grid)

        result = aligner.convert_to_target(df, ['x'], target)

        pd.testing.assert_series_equal(result['x'].dropna(), _series(expected), check_freq=False)

    @pytest.mark.parametrize('grid, values, target, expected', _AGGREGATION_CASES)
    def test_index_is_unchanged(self, aligner, grid, values, target, expected):
        """Aggregation never adds, drops nor reorders rows."""
        df = _frame(values, *grid)

        result = aligner.convert_to_target(df, ['x'], target)

        pd.testing.assert_index_equal(result.index, df.index)


class TestAggregationMethods:
    """``agg_method`` accepts every method of ``aggregate_to_lower_frequency``, ``'sum'`` by default."""

    @pytest.mark.parametrize('agg_method, expected', [
        # T1 = {1, 2, 3}, T2 = {4, 5, 6} : valeurs d'or calculées à la main
        pytest.param('sum', [6, 15], id='sum'),
        pytest.param('mean', [2, 5], id='mean'),
        pytest.param('first', [1, 4], id='first'),
        pytest.param('last', [3, 6], id='last'),
        pytest.param('min', [1, 4], id='min'),
        pytest.param('max', [3, 6], id='max'),
        pytest.param('median', [2, 5], id='median'),
        # Écart-type d'échantillon de {1, 2, 3} : racine((1 + 0 + 1) / 2) = 1
        pytest.param('std', [1, 1], id='std'),
        pytest.param('count', [3, 3], id='count'),
    ])
    def test_numeric_methods(self, aligner, agg_method, expected):
        """Each numeric method reduces the three months of each quarter as expected."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'QS', agg_method=agg_method)

        pd.testing.assert_series_equal(
            result['x'].dropna(),
            _series(dict(zip(['2023-01-01', '2023-04-01'], expected))),
            check_freq=False,
        )

    def test_default_method_is_sum(self, aligner):
        """Without ``agg_method``, sub-periods are summed."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')

        pd.testing.assert_frame_equal(
            aligner.convert_to_target(df, ['x'], 'QS'),
            aligner.convert_to_target(df, ['x'], 'QS', agg_method='sum'),
        )

    @pytest.mark.parametrize('agg_method, expected', [
        # Indicateur 1, 1, 0 | 1, 1, 1 : T1 contient un 0, T2 n'en contient pas
        pytest.param('all', [False, True], id='all'),
        pytest.param('any', [True, True], id='any'),
    ])
    def test_boolean_methods_keep_booleans(self, aligner, agg_method, expected):
        """``'all'`` / ``'any'`` write booleans (nullable ``boolean`` column, NA off the labels)."""
        df = pd.DataFrame({'flag': [1, 1, 0, 1, 1, 1]},
                          index=pd.date_range('2023-01-01', periods=6, freq='MS'))

        result = aligner.convert_to_target(df, ['flag'], 'QS', agg_method=agg_method)

        assert result['flag'].dropna().tolist() == expected

    def test_boolean_methods_raise_no_dtype_warning(self, aligner):
        """Writing booleans next to NaN does not trigger pandas' incompatible-dtype warning."""
        df = _frame([1, 1, 0, 1, 1, 1], '2023-01-01', 'MS')

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = aligner.convert_to_target(df, ['x'], 'QS', agg_method='all')

        assert result['x'].dtype == 'boolean'

    def test_boolean_column_of_a_panel_is_nullable_boolean(self, aligner):
        """Every entity converted with ``'any'``: one nullable boolean column, missing off the labels."""
        panel = _panel({'A': _frame([0, 0, 1, 0, 0, 0], '2023-01-01', 'MS'),
                        'B': _frame([0, 0, 0, 1, 1, 1], '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, ['x'], 'QS', agg_method='any')

        assert result['x'].dtype == 'boolean'
        assert result['x'].dropna().tolist() == [True, False, False, True]
        assert result['x'].isna().sum() == len(result) - 4

    def test_method_is_applied_per_entity(self, aligner):
        """On a panel, the method reduces each entity on its own rows."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7) * 10, '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, ['x'], 'QS', agg_method='mean')

        # Valeur d'or : moyennes 2 et 5 pour A, 20 et 50 pour B
        assert result['x'].dropna().tolist() == [2.0, 5.0, 20.0, 50.0]

    @pytest.mark.parametrize('keys', [['x'], []], ids=['with-keys', 'without-keys'])
    def test_unsupported_method_raises(self, aligner, keys):
        """An unknown method is rejected up front, even when there is nothing to convert."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')

        with pytest.raises(ValueError, match="Unsupported aggregation method: 'avg'"):
            aligner.convert_to_target(df, keys, 'QS', agg_method='avg')


class TestAggregationIncompletePeriods:
    """Periods missing at least one sub-period are NaN (``full_periods_only``)."""

    @pytest.mark.parametrize('start, freq, expected', [
        # Février → novembre 2023 (valeurs 1..10) : T1 privé de janvier, T4 de décembre
        # Valeur d'or : T2 = 3+4+5 = 12, T3 = 6+7+8 = 21
        pytest.param('2023-02-01', 'MS', {'2023-04-01': 12, '2023-07-01': 21}, id='start'),
        pytest.param('2023-02-28', 'ME', {'2023-06-30': 12, '2023-09-30': 21}, id='end'),
    ])
    def test_incomplete_edge_quarters_are_nan(self, aligner, start, freq, expected):
        """Truncated first and last quarters yield no value, inner quarters are summed."""
        df = _frame(np.arange(1, 11), start, freq)

        result = aligner.convert_to_target(df, ['x'], 'Q')

        pd.testing.assert_series_equal(result['x'].dropna(), _series(expected), check_freq=False)

    def test_nan_inside_a_period_masks_only_that_period(self, aligner):
        """A missing month (May) voids its quarter only."""
        # Valeurs 1..12 de 2023, mai (5) manquant : T2 incomplet
        values = np.arange(1, 13, dtype=float)
        values[4] = np.nan
        df = _frame(values, '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'QS')

        # Valeur d'or : T1 = 6, T3 = 24, T4 = 33 ; T2 absent
        expected = _series({'2023-01-01': 6, '2023-07-01': 24, '2023-10-01': 33})
        pd.testing.assert_series_equal(result['x'].dropna(), expected, check_freq=False)

    def test_holes_on_a_month_grid_mask_their_quarter_only(self, aligner):
        """Two missing months (March, August) void Q1 and Q3, Q2 and Q4 are summed."""
        # Valeurs 0..11 de 2020, mars (2) et août (7) manquants
        values = np.arange(12, dtype=float)
        values[[2, 7]] = np.nan
        df = _frame(values, '2020-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'Q')

        # Valeur d'or : T2 = 3+4+5 = 12, T4 = 9+10+11 = 30
        expected = _series({'2020-04-01': 12, '2020-10-01': 30})
        pd.testing.assert_series_equal(result['x'].dropna(), expected, check_freq=False)

    def test_quarterly_variable_on_a_month_grid_is_aggregated_as_quarterly(self, aligner):
        """A quarterly column carried on a monthly index expects 4 sub-periods per year.

        The source frequency is detected on the observed values (quarterly),
        not on the index (monthly): otherwise every year would expect 12
        values and ``full_periods_only`` would void the whole column.
        """
        # Variable trimestrielle (10 par trimestre) sur une grille mensuelle de 2 ans
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        gdp = pd.Series(np.nan, index=dates)
        gdp[dates.month.isin([1, 4, 7, 10])] = 10.0
        df = pd.DataFrame({'gdp': gdp, 'dense': 1.0})

        result = aligner.convert_to_target(df, ['gdp'], 'Y')

        # Valeur d'or : 4 trimestres × 10 = 40 au 1er janvier de chaque année
        expected = _series({'2020-01-01': 40, '2021-01-01': 40}, name='gdp')
        pd.testing.assert_series_equal(result['gdp'].dropna(), expected, check_freq=False)

    def test_one_missing_month_on_a_daily_grid_costs_only_its_year(self, aligner):
        """A monthly variable on a daily grid with one hole keeps the 8 other years.

        Frequency detection is modal: the isolated hole (May 2018) does not
        make the variable fall back to the index frequency (daily), which would
        expect 365 sub-periods per year and void the nine years.
        """
        # Variable mensuelle (1 en fin de mois) portée par une grille journalière 2015-2023
        days = pd.date_range('2015-01-01', '2023-12-31', freq='D')
        monthly = pd.Series(np.nan, index=days)
        monthly[days.is_month_end] = 1.0
        monthly['2018-05-31'] = np.nan
        df = pd.DataFrame({'x': monthly})

        result = aligner.convert_to_target(df, ['x'], 'YE')

        # Valeur d'or : 12 mois × 1 pour chaque année complète, 2018 écartée
        kept_years = [2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023]
        expected = _series({f'{year}-12-31': 12 for year in kept_years})
        pd.testing.assert_series_equal(result['x'].dropna(), expected, check_freq=False)

    def test_single_observation_on_a_dense_grid_is_an_incomplete_period(self, aligner):
        """One observation has no frequency of its own: the index grid (monthly) is assumed.

        Fallback documented in ``_aggregate_series``: with fewer than two
        observations, the converter detects the frequency on the index, so
        the lone April value is one month out of three of its quarter.
        """
        # Une seule valeur (avril) dans une colonne d'une grille mensuelle dense
        df = _frame(np.ones(12), '2023-01-01', 'MS', col='dense')
        df['x'] = np.nan
        df.loc['2023-04-01', 'x'] = 5.0

        result = aligner.convert_to_target(df, ['x'], 'QS')

        assert result['x'].isna().all()

    def test_single_row_frame_is_returned_unchanged(self, aligner):
        """A one-row frame has no frequency at all: its only value is kept."""
        df = _frame([100.0], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        pd.testing.assert_frame_equal(result, df, check_freq=False)


class TestAggregationContract:
    """Index, dtype and untouched-column guarantees of the aggregation route."""

    def test_other_columns_are_untouched(self, aligner):
        """Columns that are not converted keep their values."""
        df = _frame(np.ones(8), '2023-01-31', 'ME')
        df['y'] = np.arange(8, dtype=float)

        result = aligner.convert_to_target(df, ['x'], 'QE')

        pd.testing.assert_series_equal(result['y'], df['y'])

    def test_unsorted_input_keeps_its_row_order(self, aligner):
        """The output rows follow the (shuffled) input order."""
        shuffled = shuffle_rows(_frame(np.arange(1, 13), '2023-01-01', 'MS'), seed=1)

        result = aligner.convert_to_target(shuffled, ['x'], 'QS')

        pd.testing.assert_index_equal(result.index, shuffled.index)

    def test_unsorted_input_gives_the_sorted_values(self, aligner):
        """Shuffled rows give the same quarterly sums as sorted rows."""
        df = _frame(np.arange(1, 13), '2023-01-01', 'MS')

        result = aligner.convert_to_target(shuffle_rows(df, seed=1), ['x'], 'QS')

        pd.testing.assert_frame_equal(
            result.sort_index(), aligner.convert_to_target(df, ['x'], 'QS'), check_freq=False
        )

    def test_integer_column_is_promoted_to_float(self, aligner):
        """An integer column receives NaN, hence becomes float, with exact sums."""
        df = pd.DataFrame(
            {'x': np.arange(1, 13)}, index=pd.date_range('2023-01-01', periods=12, freq='MS')
        )

        result = aligner.convert_to_target(df, ['x'], 'QS')

        # Valeur d'or : sommes trimestrielles 6, 15, 24, 33 en flottants
        pd.testing.assert_series_equal(
            result['x'].dropna(),
            _series({'2023-01-01': 6, '2023-04-01': 15, '2023-07-01': 24, '2023-10-01': 33}),
            check_freq=False,
        )

    def test_column_without_observation_is_left_as_is(self, aligner):
        """An entirely NaN column has nothing to aggregate and stays NaN, without error."""
        df = _frame([np.nan] * 8, '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'QS')

        pd.testing.assert_frame_equal(result, df)

    def test_panel_aggregates_each_entity_on_its_own_rows(self, aligner):
        """Each entity is aggregated independently, the panel index is unchanged."""
        # Panel A / B mensuel (fin de mois) sur 6 mois : x = 1 pour A, 2 pour B
        panel = _panel({
            'A': _frame(np.ones(6), '2023-01-31', 'ME'),
            'B': _frame(np.full(6, 2.0), '2023-01-31', 'ME'),
        })

        result = aligner.convert_to_target(panel, [('A', 'x'), ('B', 'x')], 'QE')

        # Valeur d'or : 3 mois × 1 = 3 pour A, 3 mois × 2 = 6 pour B, aux fins de trimestre
        expected = pd.Series(
            [3.0, 3.0, 6.0, 6.0],
            index=pd.MultiIndex.from_product(
                [['A', 'B'], pd.DatetimeIndex(['2023-03-31', '2023-06-30'])],
                names=['entity', 'date'],
            ),
            name='x',
        )
        pd.testing.assert_series_equal(result['x'].dropna(), expected)

    def test_panel_with_positionless_target_keeps_every_row(self, aligner):
        """A positionless target ('Q') on a month-start panel lands on the existing rows."""
        panel = _panel({'A': _frame(np.ones(8), '2023-01-01', 'MS'),
                        'B': _frame(np.ones(8), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x'), ('B', 'x')], 'Q')

        # Valeur d'or : T1 et T2 complets (3 mois × 1), T3 tronqué (2 mois) → NaN
        expected = pd.Series(
            3.0,
            index=pd.MultiIndex.from_product(
                [['A', 'B'], pd.DatetimeIndex(['2023-01-01', '2023-04-01'])],
                names=['entity', 'date'],
            ),
            name='x',
        )
        pd.testing.assert_series_equal(result['x'].dropna(), expected)

    def test_panel_entity_without_key_is_untouched(self, aligner):
        """Only the keyed entity is converted, the others keep their values."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x')], 'QS')

        pd.testing.assert_series_equal(result.loc['B', 'x'], panel.loc['B', 'x'])

    def test_duplicated_dates_are_rejected(self, aligner):
        """Duplicated dates raise instead of being summed twice."""
        # Doublon de janvier ajouté en queue : la somme de T1 deviendrait 1+2+3+1 = 7
        df = with_duplicated_rows(_frame(np.arange(1, 7), '2023-01-01', 'MS'), n=1)

        with pytest.raises(ValueError, match='Duplicate dates: 2023-01-01'):
            aligner.convert_to_target(df, ['x'], 'QS')

    def test_duplicated_dates_of_a_panel_entity_name_the_entity(self, aligner):
        """On a panel, the error names the entity holding the duplicates."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': with_duplicated_rows(_frame(np.arange(1, 7), '2023-01-01', 'MS'))})

        with pytest.raises(ValueError, match=r"Duplicate dates for entity \('B',\)"):
            aligner.convert_to_target(panel, ['x'], 'QS')

    def test_duplicates_of_an_entity_without_key_are_tolerated(self, aligner):
        """Only the entities holding a column to convert are checked."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': with_duplicated_rows(_frame(np.arange(1, 7), '2023-01-01', 'MS'))})

        result = aligner.convert_to_target(panel, [('A', 'x')], 'QS')

        # Valeur d'or : T1 = 6, T2 = 15 pour A ; B intacte
        entity_a = result.index.get_level_values('entity') == 'A'
        assert result.loc[entity_a, 'x'].dropna().tolist() == [6.0, 15.0]
