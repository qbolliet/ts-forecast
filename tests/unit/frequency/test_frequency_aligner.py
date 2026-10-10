"""Tests for tsforecast.frequency.frequency_aligner.

``FrequencyAligner`` is a standalone alignment tool (it is not used by
``HighFrequencyImputer``): it brings selected variables of a time series or a
panel to a target frequency, aggregating (sum) the variables finer than the
target and interpolating the coarser ones. Its index contracts are:

- the output index is the input index extended by the target dates it lacks:
  aggregation onto labels already present keeps it untouched (same rows, same
  order); interpolation adds the target grid covering the **full source
  periods** of the first and last observations;
- a converted column lives on the target grid only: NaN elsewhere, and NaN
  for incomplete aggregated periods (``full_periods_only``);
- an explicit target position is honoured, a position-less target follows the
  position of the source index;
- the source frequency is a property of the (entity, column) pair, detected
  on the observed values, not on the index.

Covered symbols, through the public API only (``convert_to_target``,
``build_densified_index``): M→Q, M→Y, Q→Y aggregation with every
``agg_method``, Y→Q, Y→M, Q→M interpolation and the edge behaviour of each
``interp_method``, start and end positions on both sides, incomplete periods,
NaN inside a period, interpolation parameters, per-entity targets, plain
column names on a panel, absent entities and columns, duplicated dates,
non-datetime time axes, panels with a frequency per (entity, column)
(``heterogeneous_coverage_panel``, ``depenses_publiques_pib``), shuffled rows,
special column names, three-level index, ``irregular_index_timeseries``.
The defensive guards of the private helpers, unreachable from the public API,
are grouped in ``TestPrivateGuards`` (marker ``internal``).
"""
# Modules de base
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.frequency_aligner import FrequencyAligner

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

# Entités du panel réaliste à couvertures hétérogènes
_COUNTRIES = ['France', 'Allemagne', 'Italie']


@pytest.fixture
def aligner() -> FrequencyAligner:
    """Fresh FrequencyAligner instance."""
    return FrequencyAligner()


def _frame(values: List[float], start: str, freq: str, col: str = 'x') -> pd.DataFrame:
    """Build a one-column float frame on a regular grid starting at ``start``."""
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({col: np.asarray(values, dtype=float)}, index=index)


def _panel(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack one frame per entity into an (entity, date) panel."""
    panel = pd.concat(frames)
    panel.index.names = ['entity', 'date']
    return panel


def _series(mapping: Dict, name: str = 'x') -> pd.Series:
    """Build a date-indexed float series of golden values from a {date: value} mapping."""
    return pd.Series(
        list(mapping.values()), index=pd.DatetimeIndex(list(mapping)), dtype=float, name=name
    )


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
        """``'all'`` / ``'any'`` write booleans (object column, NaN off the labels)."""
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

        assert result['x'].dtype == object

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
        """Duplicated dates raise instead of being summed twice (ANO-FREQ-010)."""
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


# ------------------------------------------------------------------ #
#  Interpolation : valeurs d'or                                       #
# ------------------------------------------------------------------ #

# Cas d'interpolation : (grille source, valeurs, cible, grille attendue, valeurs d'or)
_INTERPOLATION_CASES = [
    pytest.param(
        ('2021-01-01', 'YS'), [12, 24, 36], 'MS',
        ('2021-01-01', '2023-12-01', 'MS'),
        # Valeur d'or : +1 par mois de 12 à 36, puis 36 maintenu sur 2023 (prolongement
        # vers l'avant, limite par défaut = 12 mois)
        list(range(12, 37)) + [36] * 11,
        id='Y-start->M',
    ),
    pytest.param(
        ('2021-12-31', 'YE'), [12, 24, 36], 'ME',
        ('2021-01-31', '2023-12-31', 'ME'),
        # Valeur d'or : 12 maintenu sur 2021 (prolongement vers l'arrière), puis +1 par mois
        [12] * 11 + list(range(12, 37)),
        id='Y-end->M',
    ),
    pytest.param(
        ('2023-01-01', 'QS'), [100, 130, 160], 'MS',
        ('2023-01-01', '2023-09-01', 'MS'),
        # Valeur d'or : +10 par mois, 160 maintenu sur août et septembre
        [100, 110, 120, 130, 140, 150, 160, 160, 160],
        id='Q-start->M',
    ),
    pytest.param(
        ('2023-03-31', 'QE'), [100, 130, 160], 'ME',
        ('2023-01-31', '2023-09-30', 'ME'),
        # Valeur d'or : 100 maintenu sur janvier et février, puis +10 par mois
        [100, 100, 100, 110, 120, 130, 140, 150, 160],
        id='Q-end->M',
    ),
    pytest.param(
        ('2021-01-01', 'YS'), [12, 24, 36], 'QS',
        ('2021-01-01', '2023-10-01', 'QS'),
        # Valeur d'or : +3 par trimestre, 36 maintenu sur les trois derniers trimestres
        [12, 15, 18, 21, 24, 27, 30, 33, 36, 36, 36, 36],
        id='Y-start->Q',
    ),
]


class TestInterpolationGoldenValues:
    """Upsampling interpolates linearly on the target grid covering the full source periods."""

    @pytest.mark.parametrize('grid, values, target, expected_grid, expected', _INTERPOLATION_CASES)
    def test_linear_values_on_the_target_grid(
        self, aligner, grid, values, target, expected_grid, expected
    ):
        """Interpolated values match the hand-computed linear ramp, edges held constant."""
        df = _frame(values, *grid)

        result = aligner.convert_to_target(df, ['x'], target)

        expected_series = pd.Series(
            np.asarray(expected, dtype=float),
            index=pd.date_range(expected_grid[0], expected_grid[1], freq=expected_grid[2]),
            name='x',
        )
        pd.testing.assert_series_equal(result['x'], expected_series, check_freq=False)

    def test_time_method_weights_by_calendar_days(self, aligner):
        """``interp_method='time'`` weights by the actual month lengths."""
        df = _frame([100, 130, 160], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'MS', interp_method='time')

        # Valeur d'or : 1er fév. = 100 + 30 × 31/90, 1er mars = 100 + 30 × 59/90,
        # 1er mai = 130 + 30 × 30/91, 1er juin = 130 + 30 × 61/91
        expected = [100, 100 + 30 * 31 / 90, 100 + 30 * 59 / 90, 130,
                    130 + 30 * 30 / 91, 130 + 30 * 61 / 91, 160, 160, 160]
        np.testing.assert_allclose(result['x'].to_numpy(), expected)

    @pytest.mark.parametrize('kwargs, expected', [
        # Limite d'une valeur : un seul NaN comblé après chaque observation
        pytest.param({'interp_limit': 1},
                     [100, 110, np.nan, 130, 140, np.nan, 160, 160, np.nan], id='limit-1'),
        # Zone intérieure : aucun prolongement après la dernière observation
        pytest.param({'interp_limit_area': 'inside'},
                     [100, 110, 120, 130, 140, 150, 160, np.nan, np.nan], id='limit-area-inside'),
        # Direction arrière : la queue d'août et septembre n'est plus comblée
        pytest.param({'interp_limit_direction': 'backward'},
                     [100, 110, 120, 130, 140, 150, 160, np.nan, np.nan],
                     id='limit-direction-backward'),
    ])
    def test_interpolation_parameters_are_forwarded(self, aligner, kwargs, expected):
        """``interp_limit``, ``interp_limit_area`` and ``interp_limit_direction`` reach pandas."""
        df = _frame([100, 130, 160], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'MS', **kwargs)

        np.testing.assert_array_equal(result['x'].to_numpy(), np.asarray(expected, dtype=float))

    def test_isolated_missing_quarter_is_bridged_within_the_limit(self, aligner):
        """A quarterly variable on a month grid with a missing quarter keeps its own grid.

        The variable is still detected as quarterly (modal detection), so the
        default limit is 3 months: the hole (July 2020) is bridged from April
        towards October for 3 months, August and September stay NaN.
        """
        # Trimestrielle sur grille mensuelle 2020-2021, T3 2020 manquant
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        gdp = pd.Series(np.nan, index=dates)
        observed = {'2020-01-01': 100, '2020-04-01': 110, '2020-10-01': 130,
                    '2021-01-01': 140, '2021-04-01': 150, '2021-07-01': 160, '2021-10-01': 170}
        for date, value in observed.items():
            gdp[date] = value
        df = pd.DataFrame({'gdp': gdp})

        result = aligner.convert_to_target(df, ['gdp'], 'MS')

        # Valeur d'or : pas de 20 / 6 entre avril (110) et octobre (130), 3 mois comblés
        step = 20 / 6
        expected = [100, 100 + 10 / 3, 100 + 20 / 3, 110, 110 + step, 110 + 2 * step,
                    110 + 3 * step, np.nan, np.nan, 130, 130 + 10 / 3, 130 + 20 / 3]
        np.testing.assert_allclose(result.loc['2020', 'gdp'].to_numpy(), expected)

    def test_dense_index_is_not_extended(self, aligner):
        """A frame already on the target grid is only filled, not extended."""
        # Trimestrielle sur grille mensuelle dont la dernière période est complète
        df = _frame([100, np.nan, np.nan, 110, np.nan, np.nan, 120, np.nan, np.nan],
                    '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        pd.testing.assert_index_equal(result.index, df.index)

    def test_original_observations_are_kept(self, aligner):
        """The observed values survive interpolation at their own dates."""
        df = _frame([100, 110, 120], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        pd.testing.assert_series_equal(result.loc[df.index, 'x'], df['x'], check_freq=False)

    def test_other_columns_are_nan_on_created_dates(self, aligner):
        """Columns that are not interpolated are NaN on the dates created by densification."""
        df = _frame([100, 130, 160], '2023-01-01', 'QS')
        df['other'] = [1.0, 2.0, 3.0]

        result = aligner.convert_to_target(df, ['x'], 'MS')

        created = result.index.difference(df.index)
        assert result.loc[created, 'other'].isna().all()

    def test_integer_column_is_promoted_to_float(self, aligner):
        """An integer column receives fractional values and NaN, hence becomes float."""
        df = pd.DataFrame(
            {'x': [100, 130, 160]}, index=pd.date_range('2023-01-01', periods=3, freq='QS')
        )

        result = aligner.convert_to_target(df, ['x'], 'MS')

        assert result['x'].dtype == float


# ------------------------------------------------------------------ #
#  Cohérence des positions (commit 967e2ad)                           #
# ------------------------------------------------------------------ #

# Sources en début et en fin de période, avec les bornes exactes de la densification
_POSITION_SOURCES = [
    # Position début : la dernière observation couvre les mois qui la suivent
    pytest.param(('2023-01-01', 'QS'), ('2023-01-01', '2023-09-01'), 'MS', id='Q-start'),
    # Position fin : la première observation couvre les mois qui la précèdent
    pytest.param(('2023-03-31', 'QE'), ('2023-01-31', '2023-09-30'), 'ME', id='Q-end'),
    pytest.param(('2021-01-01', 'YS'), ('2021-01-01', '2023-12-01'), 'MS', id='Y-start'),
    pytest.param(('2021-12-31', 'YE'), ('2021-01-31', '2023-12-31'), 'ME', id='Y-end'),
]

# Cibles sans position, ou explicites dans la position de la source
_SAME_POSITION_TARGETS = pytest.mark.parametrize(
    'target', ['M', 'same'], ids=['target-none', 'target-same-position']
)


def _same_position(target: str, expected_freq: str) -> str:
    """Return the monthly target to use: ``'M'`` or the source position (``expected_freq``)."""
    return expected_freq if target == 'same' else target


class TestPositionCoherence:
    """A position-less target follows the source position; an explicit one is honoured."""

    @_SAME_POSITION_TARGETS
    @pytest.mark.parametrize('grid, bounds, expected_freq', _POSITION_SOURCES)
    def test_densified_grid_has_exact_bounds_and_source_position(
        self, aligner, grid, bounds, expected_freq, target
    ):
        """The densified index is the full monthly grid of the source periods, in source position."""
        df = _frame([1.0, 2.0, 3.0], *grid)

        result = aligner.convert_to_target(df, ['x'], _same_position(target, expected_freq))

        expected = pd.date_range(bounds[0], bounds[1], freq=expected_freq)
        pd.testing.assert_index_equal(result.index, expected, exact=False)

    @_SAME_POSITION_TARGETS
    @pytest.mark.parametrize('grid, bounds, expected_freq', _POSITION_SOURCES)
    def test_interpolated_values_are_not_lost_by_reindexing(
        self, aligner, grid, bounds, expected_freq, target
    ):
        """No interpolated value is lost to a start / end mismatch with the source grid."""
        df = _frame([1.0, 2.0, 3.0], *grid)

        result = aligner.convert_to_target(df, ['x'], _same_position(target, expected_freq))

        assert result['x'].notna().all()

    def test_explicit_end_target_on_a_start_source_interpolates_on_month_ends(self, aligner):
        """QS → 'ME': the values move to the month-end grid, the quarter starts become NaN."""
        df = _frame([100, 130, 160], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'ME')

        # Valeur d'or : chaque observation est reportée sur la fin du mois qui la contient
        # (1er janv. → 31 janv.), rampe de +10 par mois ; cible en fin → direction arrière
        # par défaut, août et septembre restent NaN ; les dates d'origine ne portent plus x
        month_ends = pd.date_range('2023-01-31', '2023-09-30', freq='ME')
        expected = pd.concat([
            pd.Series([100, 110, 120, 130, 140, 150, 160, np.nan, np.nan], index=month_ends),
            pd.Series(np.nan, index=df.index),
        ]).sort_index().rename('x')
        pd.testing.assert_series_equal(result['x'], expected, check_freq=False)

    def test_explicit_start_target_on_an_end_source_interpolates_on_month_starts(self, aligner):
        """QE → 'MS': the values move to the month-start grid, the quarter ends become NaN."""
        df = _frame([100, 130, 160], '2023-03-31', 'QE')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        # Valeur d'or : 31 mars → 1er mars, rampe de +10 par mois ; cible en début →
        # direction avant par défaut, janvier et février restent NaN
        month_starts = pd.date_range('2023-01-01', '2023-09-01', freq='MS')
        expected = pd.concat([
            pd.Series([np.nan, np.nan, 100, 110, 120, 130, 140, 150, 160], index=month_starts),
            pd.Series(np.nan, index=df.index),
        ]).sort_index().rename('x')
        pd.testing.assert_series_equal(result['x'], expected, check_freq=False)

    @pytest.mark.parametrize('source, target, labels', [
        # Mois en début → fins de trimestre ajoutées ; mois en fin → débuts de trimestre ajoutés
        pytest.param(('2023-01-01', 'MS'), 'QE', ['2023-03-31', '2023-06-30'], id='M-start->Q-end'),
        pytest.param(('2023-01-31', 'ME'), 'QS', ['2023-01-01', '2023-04-01'], id='M-end->Q-start'),
    ])
    def test_explicit_opposite_target_adds_the_aggregated_labels(self, aligner, source, target, labels):
        """Aggregated labels absent from the index are added; the column lives on them only."""
        df = _frame(np.arange(1, 7), *source)
        df['other'] = np.arange(1.0, 7.0)

        result = aligner.convert_to_target(df, ['x'], target)

        # Valeur d'or : T1 = 1+2+3 = 6, T2 = 4+5+6 = 15 sur les labels ajoutés, x NaN ailleurs
        expected = pd.concat([
            _series(dict(zip(labels, [6, 15]))), pd.Series(np.nan, index=df.index),
        ]).sort_index().rename('x')
        pd.testing.assert_series_equal(result['x'], expected, check_freq=False)

    def test_explicit_opposite_target_leaves_other_columns_in_place(self, aligner):
        """The other columns keep their values and are NaN on the added labels."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')
        df['other'] = np.arange(1.0, 7.0)

        result = aligner.convert_to_target(df, ['x'], 'QE')

        pd.testing.assert_series_equal(result['other'].dropna(), df['other'], check_freq=False)

    def test_each_entity_follows_its_own_position_without_target_position(self, aligner):
        """With 'Q', a month-start entity and a month-end entity keep their own index."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7), '2023-01-31', 'ME')})

        result = aligner.convert_to_target(panel, ['x'], 'Q')

        pd.testing.assert_index_equal(result.index, panel.index)


# Comportement en bord selon la méthode d'interpolation : (méthode, valeurs d'or)
_INTERP_EDGE_CASES = [
    # Méthodes linéaires (numpy.interp) : dernière observation maintenue sur août et septembre
    pytest.param('linear', [100, 110, 120, 130, 140, 150, 160, 160, 160], id='linear'),
    pytest.param('time', [100, 100 + 30 * 31 / 90, 100 + 30 * 59 / 90, 130,
                          130 + 30 * 30 / 91, 130 + 30 * 61 / 91, 160, 160, 160], id='time'),
    # Méthodes scipy (interp1d, fill_value=NaN) : pas d'extrapolation au-delà du 1er juillet
    pytest.param('slinear', [100, 100 + 30 * 31 / 90, 100 + 30 * 59 / 90, 130,
                             130 + 30 * 30 / 91, 130 + 30 * 61 / 91, 160, np.nan, np.nan],
                 id='slinear'),
    pytest.param('zero', [100, 100, 100, 130, 130, 130, 160, np.nan, np.nan], id='zero'),
]


class TestInterpolationMethodEdges:
    """Beyond the last observation, the edge behaviour depends on ``interp_method``."""

    @pytest.mark.parametrize('interp_method, expected', _INTERP_EDGE_CASES)
    def test_edges_follow_the_method(self, aligner, interp_method, expected):
        """Linear-family methods hold the edge value, scipy methods leave the edge NaN."""
        df = _frame([100, 130, 160], '2023-01-01', 'QS')

        result = aligner.convert_to_target(df, ['x'], 'MS', interp_method=interp_method)

        np.testing.assert_allclose(result['x'].to_numpy(), np.asarray(expected, dtype=float))


# ------------------------------------------------------------------ #
#  build_densified_index                                              #
# ------------------------------------------------------------------ #

class TestBuildDensifiedIndex:
    """Union, entity by entity, of the original dates and the interpolated dates."""

    @pytest.mark.parametrize('source, interpolated_grid', [
        # Trimestres en début : la grille interpolée va du 1er janvier au 1er septembre
        pytest.param(('2023-01-01', 'QS'), ('2023-01-01', '2023-09-01', 'MS'), id='start'),
        # Trimestres en fin : la grille interpolée va du 31 janvier au 30 septembre
        pytest.param(('2023-03-31', 'QE'), ('2023-01-31', '2023-09-30', 'ME'), id='end'),
    ])
    def test_time_series_index_is_the_union(self, aligner, source, interpolated_grid):
        """For a time series, the result is the union of the source and interpolated dates."""
        df = _frame([1.0, 2.0, 3.0], *source)
        grid = pd.date_range(interpolated_grid[0], interpolated_grid[1], freq=interpolated_grid[2])
        interpolated = {(): {'x': pd.Series(0.0, index=grid)}}

        result = aligner.build_densified_index(df, interpolated, is_panel=False)

        pd.testing.assert_index_equal(result, grid, exact=False)

    def test_source_dates_outside_the_interpolated_grid_are_kept(self, aligner):
        """Original dates not on the interpolated grid stay in the union (bounds widened)."""
        # Ancre annuelle isolée de 2015 avant une grille interpolée 2018
        df = pd.DataFrame({'x': [1.0, 2.0]},
                          index=pd.DatetimeIndex(['2015-01-01', '2018-01-01']))
        grid = pd.date_range('2018-01-01', '2018-03-01', freq='MS')
        interpolated = {(): {'x': pd.Series(0.0, index=grid)}}

        result = aligner.build_densified_index(df, interpolated, is_panel=False)

        expected = pd.DatetimeIndex(['2015-01-01', '2018-01-01', '2018-02-01', '2018-03-01'])
        pd.testing.assert_index_equal(result, expected, exact=False)

    def test_panel_densifies_each_entity_on_its_own_dates(self, aligner):
        """Each entity gets the union of its own dates and its own interpolated dates."""
        # A en début de trimestre, B en fin : deux grilles mensuelles différentes ;
        # niveaux d'index aux noms non standards, conservés
        panel = pd.concat({
            'A': _frame([1.0, 2.0], '2023-01-01', 'QS'),
            'B': _frame([1.0, 2.0], '2023-03-31', 'QE'),
        })
        panel.index.names = ['pays', 'periode']
        grid_a = pd.date_range('2023-01-01', '2023-06-01', freq='MS')
        grid_b = pd.date_range('2023-01-31', '2023-06-30', freq='ME')
        interpolated = {('A',): {'x': pd.Series(0.0, index=grid_a)},
                        ('B',): {'x': pd.Series(0.0, index=grid_b)}}

        result = aligner.build_densified_index(panel, interpolated, is_panel=True)

        expected = pd.MultiIndex.from_tuples(
            [('A', date) for date in grid_a] + [('B', date) for date in grid_b],
            names=['pays', 'periode'],
        )
        pd.testing.assert_index_equal(result, expected)

    def test_entity_without_interpolation_keeps_its_dates(self, aligner):
        """An entity absent from ``interpolated`` keeps exactly its original dates."""
        panel = _panel({'A': _frame([1.0, 2.0], '2023-01-01', 'QS'),
                        'B': _frame([1.0, 2.0], '2023-01-01', 'QS')})
        grid_a = pd.date_range('2023-01-01', '2023-06-01', freq='MS')
        interpolated = {('A',): {'x': pd.Series(0.0, index=grid_a)}}

        result = aligner.build_densified_index(panel, interpolated, is_panel=True)

        pd.testing.assert_index_equal(
            result[result.get_level_values('entity') == 'B'], panel.loc[['B']].index
        )

    def test_empty_panel_returns_the_original_index(self, aligner):
        """A panel without any row has no entity to densify: its index is returned."""
        empty = _panel({'A': _frame([1.0], '2023-01-01', 'QS')}).iloc[0:0]

        result = aligner.build_densified_index(empty, {}, is_panel=True)

        pd.testing.assert_index_equal(result, empty.index)

    def test_three_level_index_uses_entity_tuples(self, aligner):
        """With a (region, entity, date) index, entities are keyed by 2-tuples."""
        panel = to_three_level_index(_panel({'A': _frame([1.0, 2.0], '2023-01-01', 'QS')}))
        grid = pd.date_range('2023-01-01', '2023-06-01', freq='MS')
        interpolated = {('Zone euro', 'A'): {'x': pd.Series(0.0, index=grid)}}

        result = aligner.build_densified_index(panel, interpolated, is_panel=True)

        expected = pd.MultiIndex.from_tuples(
            [('Zone euro', 'A', date) for date in grid], names=['region', 'entity', 'date']
        )
        pd.testing.assert_index_equal(result, expected)


# ------------------------------------------------------------------ #
#  convert_to_target : orientation et clés                            #
# ------------------------------------------------------------------ #

class TestConvertToTargetRouting:
    """Each key is aggregated or interpolated according to its own observed frequency."""

    def test_mixed_keys_are_routed_independently(self, aligner):
        """A monthly and a yearly column are respectively aggregated and interpolated to Q."""
        # Mensuelle 1..24 et annuelle (40 en 2022, 80 en 2023) sur une grille mensuelle
        df = _frame(np.arange(1, 25), '2022-01-01', 'MS', col='m')
        df['y'] = np.nan
        df.loc['2022-01-01', 'y'] = 40.0
        df.loc['2023-01-01', 'y'] = 80.0

        result = aligner.convert_to_target(df, ['m', 'y'], 'QS')

        # Valeur d'or : m = sommes trimestrielles (6, 15, ..., 69) ; y = rampe de +10 par
        # trimestre de 40 à 80, puis 80 maintenu
        expected = pd.DataFrame(
            {'m': [6.0, 15.0, 24.0, 33.0, 42.0, 51.0, 60.0, 69.0],
             'y': [40.0, 50.0, 60.0, 70.0, 80.0, 80.0, 80.0, 80.0]},
            index=pd.date_range('2022-01-01', periods=8, freq='QS'),
        )
        pd.testing.assert_frame_equal(result.dropna(how='all'), expected, check_freq=False)

    def test_same_frequency_leaves_values_unchanged(self, aligner):
        """A monthly column converted to monthly is a sum of one sub-period: identity."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        pd.testing.assert_frame_equal(result, df)

    def test_empty_keys_return_the_frame_unchanged(self, aligner):
        """No key to convert: the frame is returned as is."""
        df = _frame(np.ones(4), '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, [], 'QS')

        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize('grid, values, target', [
        pytest.param(('2023-01-01', 'MS'), np.ones(4), 'QS', id='aggregation'),
        pytest.param(('2023-01-01', 'QS'), [100, 110, 120], 'MS', id='interpolation'),
    ])
    def test_missing_column_is_ignored(self, aligner, grid, values, target):
        """A key naming an absent column is skipped, not an error."""
        df = _frame(values, *grid)

        result = aligner.convert_to_target(df, ['missing'], target)

        pd.testing.assert_frame_equal(result, df)

    def test_panel_missing_column_is_ignored(self, aligner):
        """Same in a panel: an (entity, absent column) key is skipped."""
        panel = _panel({'A': _frame([100, 110, 120], '2023-01-01', 'QS')})

        result = aligner.convert_to_target(panel, [('A', 'missing')], 'MS')

        pd.testing.assert_frame_equal(result, panel)

    def test_unknown_entity_is_ignored(self, aligner):
        """A key naming an entity absent from the panel is skipped, like an absent column."""
        panel = _panel({'A': _frame(np.ones(6), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('Z', 'x')], 'QS')

        pd.testing.assert_frame_equal(result, panel)

    def test_panel_column_without_observation_is_left_as_is(self, aligner):
        """An entity whose column is entirely NaN is left untouched, without error."""
        panel = _panel({'A': _frame([np.nan] * 8, '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x')], 'QS')

        pd.testing.assert_frame_equal(result, panel)

    def test_per_entity_target_missing_an_entity_raises(self, aligner):
        """A target dict without one of the keyed entities raises a ValueError naming it."""
        panel = _panel({'A': _frame(np.ones(6), '2023-01-01', 'MS'),
                        'B': _frame(np.ones(6), '2023-01-01', 'MS')})

        with pytest.raises(ValueError, match="'B'"):
            aligner.convert_to_target(panel, [('A', 'x'), ('B', 'x')], {'A': 'QS'})

    def test_duplicated_dates_raise_on_interpolation(self, aligner):
        """Duplicated dates are rejected by the interpolation route, with the same message."""
        df = with_duplicated_rows(_frame([100, 130, 160], '2023-01-01', 'QS'), n=1)

        with pytest.raises(ValueError, match='Duplicate dates: 2023-01-01'):
            aligner.convert_to_target(df, ['x'], 'MS')

    def test_empty_frame_is_returned_unchanged(self, aligner):
        """A frame without rows has nothing to convert."""
        df = _frame([1.0], '2023-01-01', 'MS').iloc[0:0]

        result = aligner.convert_to_target(df, ['x'], 'QS')

        pd.testing.assert_frame_equal(result, df)

    def test_plain_column_name_on_a_panel_converts_every_entity(self, aligner):
        """A bare column name on a panel designates the column of every entity (ANO-FREQ-011)."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7) * 2, '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, ['x'], 'QS')

        # Valeur d'or : T1 = 6, T2 = 15 pour A ; le double pour B
        assert result['x'].dropna().tolist() == [6.0, 15.0, 12.0, 30.0]

    def test_plain_and_tuple_keys_are_equivalent(self, aligner, heterogeneous_coverage_panel):
        """A column name gives the same result as its (entity, column) keys, per-entity frequencies included."""
        tuple_keys = [(country, 'depenses_publiques_pib') for country in _COUNTRIES]

        result = aligner.convert_to_target(
            heterogeneous_coverage_panel, ['depenses_publiques_pib'], 'QS'
        )

        pd.testing.assert_frame_equal(
            result, aligner.convert_to_target(heterogeneous_coverage_panel, tuple_keys, 'QS')
        )

    def test_several_plain_column_names_on_a_panel(self, aligner):
        """Several bare column names are each expanded to every entity."""
        frame = _frame(np.arange(1, 7), '2023-01-01', 'MS')
        frame['y'] = np.arange(1, 7) * 10.0
        panel = _panel({'A': frame, 'B': frame})

        result = aligner.convert_to_target(panel, ['x', 'y'], 'QS')

        # Valeur d'or : x = 6, 15 et y = 60, 150 pour chacune des deux entités
        expected = pd.DataFrame({'x': [6.0, 15.0] * 2, 'y': [60.0, 150.0] * 2},
                                index=result.dropna().index)
        pd.testing.assert_frame_equal(result.dropna(), expected)

    def test_plain_and_tuple_keys_for_the_same_column_are_merged(self, aligner):
        """Mixing ``'x'`` and ``('A', 'x')`` converts each entity once."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x'), 'x'], 'QS')

        pd.testing.assert_frame_equal(result, aligner.convert_to_target(panel, ['x'], 'QS'))

    @pytest.mark.parametrize('index_kind', ['period', 'range'])
    def test_non_datetime_time_series_index_raises(self, aligner, index_kind):
        """A PeriodIndex (or any non-datetime index) is rejected with an explicit message."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')
        df = df.to_period('M') if index_kind == 'period' else df.reset_index(drop=True)

        with pytest.raises(TypeError, match='requires a DatetimeIndex as the index'):
            aligner.convert_to_target(df, ['x'], 'QS')

    def test_period_index_message_suggests_the_conversion(self, aligner):
        """The PeriodIndex message points to ``to_timestamp``."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS').to_period('M')

        with pytest.raises(TypeError, match=r"to_timestamp\(how='start'\)"):
            aligner.convert_to_target(df, ['x'], 'QS')

    def test_period_level_of_a_panel_raises(self, aligner):
        """On a panel, the message names the last level of the MultiIndex."""
        panel = pd.concat({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS').to_period('M')})

        with pytest.raises(TypeError, match='last level of the panel MultiIndex, got PeriodIndex'):
            aligner.convert_to_target(panel, [('A', 'x')], 'QS')


# ------------------------------------------------------------------ #
#  Panel à fréquence par (entité, colonne) : jeu réaliste             #
# ------------------------------------------------------------------ #

# Dépenses publiques : annuelles pour France et Italie, trimestrielles pour Allemagne
_DEPENSES_KEYS = [(country, 'depenses_publiques_pib') for country in _COUNTRIES]


class TestHeterogeneousCoveragePanel:
    """Frequency is a property of the (entity, column) pair (``heterogeneous_coverage_panel``).

    ``depenses_publiques_pib`` is yearly for France and Italie, quarterly for
    Allemagne; ``climat_affaires`` is monthly for France and Allemagne,
    structurally absent for Italie; every entity has its own coverage.
    """

    def test_quarterly_target_keeps_the_panel_index(self, aligner, heterogeneous_coverage_panel):
        """Interpolating yearly entities to quarter starts adds no row (already on the grid)."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        pd.testing.assert_index_equal(result.index, heterogeneous_coverage_panel.index)

    def test_quarterly_entity_is_unchanged_by_a_quarterly_target(
        self, aligner, heterogeneous_coverage_panel
    ):
        """Allemagne is already quarterly: Q→Q is the identity for it."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        pd.testing.assert_series_equal(
            result.loc['Allemagne', 'depenses_publiques_pib'],
            heterogeneous_coverage_panel.loc['Allemagne', 'depenses_publiques_pib'],
        )

    @pytest.mark.parametrize('country', ['France', 'Italie'])
    def test_yearly_entities_are_interpolated_quarterly(
        self, aligner, heterogeneous_coverage_panel, country
    ):
        """Each quarter start of a yearly entity is a quarter of the way to the next year."""
        source = heterogeneous_coverage_panel.loc[country, 'depenses_publiques_pib'].dropna()

        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        # Valeur d'or : v(an) + k/4 × (v(an+1) − v(an)) au k-ième début de trimestre,
        # puis dernière valeur maintenue sur les trois trimestres suivants (limite 4)
        expected = {}
        next_values = list(source.to_numpy()[1:]) + [None]
        for (date, value), next_value in zip(source.items(), next_values):
            step = 0.0 if next_value is None else (next_value - value) / 4
            for k in range(4):
                expected[date + pd.DateOffset(months=3 * k)] = value + k * step
        np.testing.assert_allclose(
            result.loc[country, 'depenses_publiques_pib'].dropna().to_numpy(),
            _series(expected).to_numpy(),
        )

    def test_yearly_entities_stay_nan_outside_quarter_starts(
        self, aligner, heterogeneous_coverage_panel
    ):
        """Months that are not quarter starts, and the withdrawn last publication, stay NaN."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        france = result.loc['France', 'depenses_publiques_pib']
        # Hors débuts de trimestre, et après le dernier trimestre couvert (janvier 2024
        # retiré du jeu) : aucune valeur inventée
        outside = ~france.index.month.isin([1, 4, 7, 10]) | (france.index >= '2024-01-01')
        assert france[outside].isna().all()

    def test_per_entity_target_aggregates_only_complete_years(
        self, aligner, heterogeneous_coverage_panel
    ):
        """With a target dict, Allemagne (Q→Y) sums its complete years only."""
        targets = {'France': 'QS', 'Allemagne': 'YS', 'Italie': 'QS'}
        quarterly = heterogeneous_coverage_panel.loc['Allemagne', 'depenses_publiques_pib'].dropna()

        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, targets)

        # Valeur d'or : somme des 4 trimestres de chaque année complète (2019-2023) ;
        # 2018 (T3-T4 seulement) et 2024 (T1 retiré) sont incomplètes
        by_year = quarterly.groupby(quarterly.index.year)
        complete = by_year.sum()[by_year.count() == 4]
        expected = pd.Series(
            complete.to_numpy(),
            index=pd.DatetimeIndex([f'{year}-01-01' for year in complete.index], name='date'),
            name='depenses_publiques_pib',
        )
        pd.testing.assert_series_equal(
            result.loc['Allemagne', 'depenses_publiques_pib'].dropna(), expected
        )

    def test_structurally_absent_column_is_left_untouched(self, aligner, heterogeneous_coverage_panel):
        """``climat_affaires`` of Italie (zero observation) stays NaN, without error."""
        keys = [(country, 'climat_affaires') for country in _COUNTRIES]

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'QS')

        assert result.loc['Italie', 'climat_affaires'].isna().all()

    def test_monthly_entities_are_summed_by_quarter(self, aligner, heterogeneous_coverage_panel):
        """``climat_affaires`` of France is summed over the three months of each quarter."""
        keys = [(country, 'climat_affaires') for country in _COUNTRIES]
        monthly = heterogeneous_coverage_panel.loc['France', 'climat_affaires']

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'QS')

        # Valeur d'or : T1 2019 = janvier + février + mars 2019
        expected = monthly['2019-01-01':'2019-03-01'].sum()
        assert result.loc[('France', pd.Timestamp('2019-01-01')), 'climat_affaires'] == (
            pytest.approx(expected)
        )

    @pytest.mark.parametrize('country, first, last', [
        # Bornes : première ancre annuelle (historique antérieur à la grille mensuelle)
        # et dernière date de la grille mensuelle de l'entité
        pytest.param('France', '2015-01-01', '2024-07-01', id='France'),
        pytest.param('Allemagne', '2016-01-01', '2024-04-01', id='Allemagne'),
        pytest.param('Italie', '2016-01-01', '2024-07-01', id='Italie'),
    ])
    def test_monthly_densification_respects_each_entity_bounds(
        self, aligner, heterogeneous_coverage_panel, country, first, last
    ):
        """Yearly ``balance_commerciale_annuelle`` → MS fills each entity's own span, monthly."""
        keys = [(entity, 'balance_commerciale_annuelle') for entity in _COUNTRIES]

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'MS')

        pd.testing.assert_index_equal(
            result.loc[country].index,
            pd.date_range(first, last, freq='MS', name='date'),
            exact=False,
        )


# ------------------------------------------------------------------ #
#  Robustesse : désordre, noms, niveaux d'index, index irrégulier     #
# ------------------------------------------------------------------ #

# Clés mélangeant agrégation (climat_affaires → Q) et interpolation (annuelles → Q)
_MIXED_PANEL_KEYS = (
    [(country, 'climat_affaires') for country in _COUNTRIES]
    + _DEPENSES_KEYS
    + [(country, 'balance_commerciale_annuelle') for country in _COUNTRIES]
)


class TestRobustness:
    """The result does not depend on row order, column names, index names or index depth."""

    @pytest.fixture
    def reference(self, aligner, heterogeneous_coverage_panel) -> pd.DataFrame:
        """Conversion of the sorted panel, reference of the perturbed variants."""
        return aligner.convert_to_target(heterogeneous_coverage_panel, _MIXED_PANEL_KEYS, 'QS')

    @pytest.mark.parametrize('seed', [0, 7])
    def test_shuffled_panel_gives_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference, seed
    ):
        """Shuffling the panel rows gives the same values once sorted back."""
        result = aligner.convert_to_target(
            shuffle_rows(heterogeneous_coverage_panel, seed=seed), _MIXED_PANEL_KEYS, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_reversed_entities_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """The order of entity blocks has no effect."""
        result = aligner.convert_to_target(
            reverse_entities(heterogeneous_coverage_panel), _MIXED_PANEL_KEYS, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    @pytest.mark.parametrize('seed', [0, 7])
    def test_shuffled_time_series_gives_the_same_result(
        self, aligner, irregular_index_timeseries, seed
    ):
        """Shuffling a time series (aggregation and interpolation keys) has no effect."""
        keys = ['production_industrielle', 'balance_commerciale_annuelle']
        expected = aligner.convert_to_target(irregular_index_timeseries, keys, 'QS')

        result = aligner.convert_to_target(
            shuffle_rows(irregular_index_timeseries, seed=seed), keys, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), expected.sort_index(), check_freq=False)

    def test_special_column_names_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """Spaces, accents, ``%``, ``/`` and parentheses in column names change nothing."""
        renamed, mapping = with_special_column_names(heterogeneous_coverage_panel)
        keys = [(entity, mapping[col]) for entity, col in _MIXED_PANEL_KEYS]

        result = aligner.convert_to_target(renamed, keys, 'QS')

        # Restauration des noms d'origine pour la comparaison
        restored = result.rename(columns={new: old for old, new in mapping.items()})
        pd.testing.assert_frame_equal(restored, reference)

    def test_non_standard_index_names_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """Levels named neither ``entity`` nor ``date`` are handled by position."""
        renamed = with_index_names(heterogeneous_coverage_panel, ['pays', 'periode'])

        result = aligner.convert_to_target(renamed, _MIXED_PANEL_KEYS, 'QS')

        pd.testing.assert_frame_equal(result, with_index_names(reference, ['pays', 'periode']))

    def test_three_level_index_gives_the_same_values(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """A (region, country, date) index with (region, country, column) keys gives the same values."""
        three_levels = to_three_level_index(heterogeneous_coverage_panel)
        keys = [('Zone euro', entity, col) for entity, col in _MIXED_PANEL_KEYS]

        result = aligner.convert_to_target(three_levels, keys, 'QS')

        pd.testing.assert_frame_equal(result, to_three_level_index(reference))

    def test_irregular_time_series_is_densified_to_a_regular_grid(
        self, aligner, irregular_index_timeseries
    ):
        """Yearly anchors before the monthly grid are bridged: the output grid is regular."""
        # Ancres annuelles 2015-2017 isolées avant la grille mensuelle 2018-01 → 2024-07
        result = aligner.convert_to_target(
            irregular_index_timeseries, ['balance_commerciale_annuelle'], 'MS'
        )

        expected = pd.date_range('2015-01-01', '2024-07-01', freq='MS', name='date')
        pd.testing.assert_index_equal(result.index, expected, exact=False)

    def test_irregular_time_series_keeps_every_observation(
        self, aligner, irregular_index_timeseries
    ):
        """No observed value of any column is altered by the densification."""
        result = aligner.convert_to_target(
            irregular_index_timeseries, ['balance_commerciale_annuelle'], 'MS'
        )

        observed = irregular_index_timeseries.stack()
        pd.testing.assert_series_equal(result.stack().loc[observed.index], observed)


# ------------------------------------------------------------------ #
#  Gardes défensives des méthodes privées                             #
# ------------------------------------------------------------------ #

@pytest.mark.internal
class TestPrivateGuards:
    """Defensive guards of the private helpers, unreachable through ``convert_to_target``.

    ``convert_to_target`` only calls ``_aggregate_to_target`` and
    ``_interpolate_to_target`` with non-empty key lists, and routes every
    column without a detectable frequency (no or one observation) to the
    aggregation: these guards cannot be reached from the public API.
    """

    @pytest.mark.parametrize('method', ['_aggregate_to_target', '_interpolate_to_target'])
    def test_empty_keys_return_the_frame_unchanged(self, aligner, method):
        """Both dataset-level helpers return their input when no key is given."""
        df = _frame([100, 110, 120], '2023-01-01', 'QS')

        result = getattr(aligner, method)(df, [], 'MS')

        assert result is df

    @pytest.mark.parametrize('values', [
        pytest.param([np.nan] * 3, id='no-observation'),
        pytest.param([100.0, np.nan, np.nan], id='single-observation'),
    ])
    def test_interpolating_a_column_without_frequency_leaves_it_unchanged(self, aligner, values):
        """Fewer than two observations: the column is left as is (ANO-FREQ-012)."""
        df = _frame(values, '2023-01-01', 'QS')

        result = aligner._interpolate_to_target(df, ['x'], 'MS')

        pd.testing.assert_frame_equal(result, df)
