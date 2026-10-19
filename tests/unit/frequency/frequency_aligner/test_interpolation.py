"""Interpolation route of ``FrequencyAligner``.

Scenarios: golden values on the target grid, position coherence, edge behaviour per
interpolation method and the densified index.
"""

import numpy as np
import pandas as pd
import pytest

# Perturbations partagées
from tests.support.perturbations import to_three_level_index

# Outils de test partagés du paquet
from tests.unit.frequency.frequency_aligner.helpers import _frame, _panel, _series


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
        # Direction arrière : la queue d'août et septembre n'est pas comblée
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
