"""Index robustness of ``CovariateMaterializer``.

The edge cases of ``CLAUDE.md``, through ``tests/support/perturbations.py``:
unsorted rows (source and grid), reversed entities, duplicated dates (grid and
source), empty and single-row data, three-level ``MultiIndex``, non-standard
index names, special column names, and period start / end positions — the
four materialization ways must land on a start-anchored grid exactly as on an
end-anchored one, with the same golden values.

Corrections of ANO-FREQ-039 (duplicated source dates rejected alike by the
four ways, the exact aggregation included) and ANO-FREQ-043 (an empty panel
grid decides like an empty time series grid) — ``tests/ANOMALIES.md``.

Covered symbols: ``CovariateMaterializer.materialize``, ``decide_ways``,
``stage_frame``.
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Jeux et perturbations partagés
from tests.support.perturbations import (
    empty_like,
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)
from tests.unit.frequency.covariate_materializer.helpers import (
    HETEROGENEOUS_FREQUENCIES,
    PANEL_FREQUENCIES,
    TS_FREQUENCIES,
    monthly_grid,
    panel_grid,
    quarterly_grid,
)


# Colonnes de TS matérialisées à l'étape Q : agrégation, identité, interpolation
TS_COLUMNS = ['m1', 'q1', 'a1']
# Colonnes du panel de référence matérialisées à l'étape M
PANEL_COLUMNS = ['m1', 'q1', 'a1', 'climat_affaires']


def _materialize(data, grid, stage, columns=TS_COLUMNS, frequencies=TS_FREQUENCIES, **kwargs):
    """Materialize ``columns`` with a fresh default materializer."""
    return CovariateMaterializer(**kwargs).materialize(
        columns=list(columns),
        grid_index=grid,
        stage_freq=stage,
        detected_frequencies=frequencies,
        source_data=data,
    )


class TestRowOrder:
    """Unsorted rows and reversed entities change neither the ways nor the values."""

    def test_shuffled_source_gives_the_same_features(self, reference_timeseries):
        """The row order of the source plays no part."""
        reference, _, _ = _materialize(reference_timeseries, quarterly_grid(), 'Q')
        shuffled, _, _ = _materialize(shuffle_rows(reference_timeseries, seed=3), quarterly_grid(), 'Q')
        pd.testing.assert_frame_equal(shuffled, reference)

    def test_shuffled_grid_is_served_by_label(self, reference_timeseries):
        """A shuffled grid gets, row by row, the values of its own dates."""
        grid = quarterly_grid()
        reference, _, _ = _materialize(reference_timeseries, grid, 'Q')
        permuted = grid[np.random.default_rng(0).permutation(len(grid))]
        shuffled, _, _ = _materialize(reference_timeseries, permuted, 'Q')
        pd.testing.assert_frame_equal(shuffled.sort_index(), reference, check_freq=False)

    @pytest.mark.parametrize(
        'perturb',
        [lambda df: shuffle_rows(df, seed=1), reverse_entities],
        ids=['shuffled', 'reversed-entities'],
    )
    def test_realistic_panel_order_changes_nothing(self, heterogeneous_coverage_panel, perturb):
        """heterogeneous_coverage_panel: same ways, same values once sorted."""
        columns = list(HETEROGENEOUS_FREQUENCIES)
        data = heterogeneous_coverage_panel
        reference, ways, _ = _materialize(data, data.index, 'M', columns, HETEROGENEOUS_FREQUENCIES)
        perturbed = perturb(data)
        result, perturbed_ways, _ = _materialize(perturbed, perturbed.index, 'M', columns, HETEROGENEOUS_FREQUENCIES)
        assert perturbed_ways == ways
        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())


class TestDuplicates:
    """Duplicated dates, on the grid and in the source."""

    def test_duplicated_grid_dates_carry_the_same_value(self, reference_timeseries):
        """A grid date listed twice gets the same value twice."""
        grid = monthly_grid()
        features, _, _ = _materialize(reference_timeseries, grid.append(grid[-1:]), 'M', ['m1', 'a1'])
        np.testing.assert_allclose(features.iloc[-2].to_numpy(), features.iloc[-1].to_numpy())

    @pytest.mark.parametrize(
        'column, grid, stage, strategy',
        [
            ('m1', monthly_grid(), 'M', 'interpolate'),
            # Agrégation exacte : sans contrôle, le T4 2023 vaudrait 537 au
            # lieu de 402 (133 + 134 + 135), décembre compté deux fois
            ('m1', quarterly_grid(), 'Q', 'interpolate'),
            ('a1', monthly_grid(), 'M', 'tolerate_nan'),
            ('a1', monthly_grid(), 'M', 'interpolate'),
        ],
        ids=['identity', 'aggregate', 'raw_anchors', 'interpolate'],
    )
    def test_duplicated_source_dates_are_rejected_by_every_way(
        self, reference_timeseries, column, grid, stage, strategy
    ):
        """The four ways refuse duplicated source dates alike, naming them (ANO-FREQ-039)."""
        # Doublon de la dernière ligne (2023-12-31, ancre de q1 et de a1)
        duplicated = with_duplicated_rows(reference_timeseries.iloc[::-1], n=1)
        with pytest.raises(ValueError, match=r"Duplicate dates in source_data: \[Timestamp\('2023-12-31"):
            _materialize(duplicated, grid, stage, [column], covariate_strategy=strategy)

    def test_duplicated_panel_pairs_name_the_entity(self, mixed_freq_panel_heterogeneous):
        """On a panel, the duplicated ``(entity, date)`` pair is named."""
        duplicated = with_duplicated_rows(mixed_freq_panel_heterogeneous, n=1)
        with pytest.raises(ValueError, match=r"\('DE', Timestamp\('2021-01-31"):
            CovariateMaterializer().stage_frame(
                grid_index=mixed_freq_panel_heterogeneous.index,
                stage_freq='M',
                detected_frequencies=PANEL_FREQUENCIES,
                source_data=duplicated,
            )


class TestEmptyAndSingle:
    """Empty grids, empty sources, a single observation."""

    def test_empty_time_series_grid(self, reference_timeseries):
        """An empty series grid gives an empty frame with the requested columns."""
        features, ways, _ = _materialize(reference_timeseries, monthly_grid()[:0], 'M', ['m1', 'a1'])
        assert (features.shape, list(features.columns), ways) == (
            (0, 2), ['m1', 'a1'], {'m1': 'identity', 'a1': 'interpolate'},
        )

    def test_empty_source_gives_nan_features(self, reference_timeseries):
        """No observation at all: every cell NaN, every column origin ``'observed'``."""
        features, _ways, origins = _materialize(empty_like(reference_timeseries), monthly_grid(), 'M', ['m1', 'a1'])
        assert (int(features.notna().sum().sum()), origins) == (0, {'m1': 'observed', 'a1': 'observed'})

    def test_single_observation_source(self, reference_timeseries):
        """One row (2021-01-31, m1 = 100): only that cell is produced."""
        features, _ways, _origins = _materialize(
            single_observation(reference_timeseries), monthly_grid(), 'M', ['m1', 'a1']
        )
        assert features.stack().to_dict() == {(pd.Timestamp('2021-01-31'), 'm1'): 100.0}

    @pytest.mark.parametrize('method', ['decide_ways', 'materialize', 'stage_frame'])
    def test_empty_panel_grid(self, mixed_freq_panel_heterogeneous, method):
        """An empty panel grid behaves like an empty series grid (ANO-FREQ-043).

        Sans entité, la décision se prend pour l'entité dégénérée ``()``,
        comme sur une série vide : une voie par colonne, et une frame vide aux
        colonnes demandées.
        """
        empty_grid = pd.MultiIndex.from_tuples([], names=['country', 'date'])
        materializer = CovariateMaterializer()
        kwargs = dict(
            columns=['m1'], grid_index=empty_grid, stage_freq='M', detected_frequencies=PANEL_FREQUENCIES,
        )
        if method == 'decide_ways':
            result = materializer.decide_ways(**kwargs)
            expected = {'m1': 'identity'}
        else:
            result = getattr(materializer, method)(source_data=mixed_freq_panel_heterogeneous, **kwargs)
            result = (result[0] if method == 'materialize' else result).shape
            expected = (0, 1)
        assert result == expected


class TestIndexShapes:
    """Three-level index, non-standard index names, special column names."""

    def test_three_level_index_gives_the_same_features(self, mixed_freq_panel_heterogeneous):
        """An outer region level changes neither the ways nor the values."""
        data = mixed_freq_panel_heterogeneous
        reference, ways, _ = _materialize(data, data.index, 'M', PANEL_COLUMNS, PANEL_FREQUENCIES)
        three = to_three_level_index(data)
        features, three_ways, _ = _materialize(three, three.index, 'M', PANEL_COLUMNS, PANEL_FREQUENCIES)
        assert three_ways == ways
        np.testing.assert_allclose(features.to_numpy(), reference.to_numpy())

    def test_non_standard_index_names(self, mixed_freq_panel_heterogeneous):
        """The levels are found by position, never by their names."""
        data = mixed_freq_panel_heterogeneous
        reference, _, _ = _materialize(data, panel_grid(data, monthly_grid()), 'M', PANEL_COLUMNS, PANEL_FREQUENCIES)
        renamed = with_index_names(data, ['pays', 'periode'])
        features, _, _ = _materialize(
            renamed, panel_grid(renamed, monthly_grid()), 'M', PANEL_COLUMNS, PANEL_FREQUENCIES
        )
        np.testing.assert_allclose(features.to_numpy(), reference.to_numpy())

    def test_special_column_names(self, reference_timeseries):
        """Spaces, accents and symbols: same values, per-feature dicts included."""
        reference, _, _ = _materialize(
            reference_timeseries, monthly_grid(), 'M', interpolation_method={'a1': 'nearest', '__default__': 'linear'}
        )
        renamed, mapping = with_special_column_names(reference_timeseries)
        features, _, _ = _materialize(
            renamed,
            monthly_grid(),
            'M',
            [mapping[column] for column in TS_COLUMNS],
            {mapping[column]: freq for column, freq in TS_FREQUENCIES.items()},
            interpolation_method={mapping['a1']: 'nearest', '__default__': 'linear'},
        )
        assert list(features.columns) == [mapping[column] for column in TS_COLUMNS]
        np.testing.assert_allclose(features.to_numpy(), reference.to_numpy())


class TestStartAnchoredGrids:
    """Grids anchored at period START (``'MS'`` / ``'QS'`` / ``'YS'``).

    The reference datasets of the spec are end-anchored; real data anchored
    at period start (``pd.date_range(..., freq='MS')``, notebooks 2 and 3)
    goes through the same ways. The labels an aggregation or an interpolation
    produces must land back on the stage grid, otherwise the final reindexing
    yields an all-NaN column — then, in cascade, an empty training set and an
    interpolation fallback.
    """

    @staticmethod
    def _start_anchored_data() -> pd.DataFrame:
        """Build a month-START anchored dataset, quarterly and annual anchors."""
        dates = pd.date_range('2021-01-01', periods=36, freq='MS')
        data = pd.DataFrame({'m1': 100.0 + np.arange(36)}, index=dates)
        data.index.name = 'date'
        # Ancres trimestrielles en DÉBUT de trimestre (janvier, avril, ...)
        data['q1'] = np.nan
        quarter_starts = dates[dates.month.isin([1, 4, 7, 10])]
        data.loc[quarter_starts, 'q1'] = 10.0 * np.arange(1, len(quarter_starts) + 1)
        # Ancres annuelles en DÉBUT d'année (janvier)
        data['a1'] = np.nan
        year_starts = dates[dates.month == 1]
        data.loc[year_starts, 'a1'] = 1000.0 * np.arange(1, len(year_starts) + 1)
        return data

    def test_aggregation_lands_on_a_quarter_start_grid(self):
        """A monthly column aggregates onto a quarter-START grid."""
        data = self._start_anchored_data()
        features, ways, _origins = CovariateMaterializer().materialize(
            columns=['m1', 'q1'],
            grid_index=pd.date_range('2021-01-01', periods=12, freq='QS'),
            stage_freq='Q',
            detected_frequencies={'m1': 'M', 'q1': 'Q', 'a1': 'Y'},
            source_data=data,
        )

        # La voie reste celle du rang 1 : c'est l'ancrage des labels qui est en jeu
        assert ways['m1'] == 'aggregate'
        # Aucune période incomplète sur cette grille : la colonne est pleine
        assert features['m1'].notna().all()
        # Somme exacte des trois mois du trimestre, non un label perdu en NaN
        assert features['m1'].iloc[0] == pytest.approx(100 + 101 + 102)
        assert features['m1'].iloc[1] == pytest.approx(103 + 104 + 105)

    def test_interpolation_lands_on_a_month_start_grid(self):
        """An annual column interpolates onto a month-START grid."""
        data = self._start_anchored_data()
        features, ways, origins = CovariateMaterializer(covariate_strategy='interpolate').materialize(
            columns=['a1'],
            grid_index=pd.date_range('2021-01-01', periods=36, freq='MS'),
            stage_freq='M',
            detected_frequencies={'m1': 'M', 'q1': 'Q', 'a1': 'Y'},
            source_data=data,
        )

        assert ways['a1'] == 'interpolate'
        assert origins['a1'] == 'interpolated'
        # La colonne est produite sur toute la grille, et non laissée vide
        assert features['a1'].notna().all()


def _positioned_dataset(position: str) -> pd.DataFrame:
    """Build the ``TS`` values anchored at period start or end.

    Args:
        position: ``'start'`` (``MS``, anchors in January / quarter-start
            months) or ``'end'`` (``ME``, anchors in December / quarter-end
            months).

    Returns:
        Frame with ``m1`` (100 + rank), ``q1`` (10 · k) and ``a1``
        (120 / 132 / 150) at the anchors of the chosen position.
    """
    dates = pd.date_range('2021-01-01' if position == 'start' else '2021-01-31', periods=36,
                          freq='MS' if position == 'start' else 'ME')
    quarter_months = [1, 4, 7, 10] if position == 'start' else [3, 6, 9, 12]
    year_month = 1 if position == 'start' else 12
    data = pd.DataFrame({'m1': 100.0 + np.arange(36), 'q1': np.nan, 'a1': np.nan}, index=dates)
    data.loc[dates[dates.month.isin(quarter_months)], 'q1'] = 10.0 * np.arange(1, 13)
    data.loc[dates[dates.month == year_month], 'a1'] = [120.0, 132.0, 150.0]
    return data


class TestStartAndEndPositions:
    """The four ways give the same golden values at period start and at period end."""

    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'column, stage, strategy, expected',
        [
            # Identité : q1 lue sur la grille trimestrielle, 10 · k
            ('q1', 'Q', 'interpolate', 10.0 * np.arange(1, 13)),
            # Agrégation : m1 sommée par trimestre, 303 + 9k
            ('m1', 'Q', 'interpolate', 303.0 + 9.0 * np.arange(12)),
            # Ancres brutes : les 3 valeurs annuelles, et elles seules
            ('a1', 'M', 'tolerate_nan', np.array([120.0, 132.0, 150.0])),
            # Interpolation : les 3 ancres traversées telles quelles
            ('a1', 'M', 'interpolate', np.array([120.0, 132.0, 150.0])),
        ],
        ids=['identity', 'aggregate', 'raw_anchors', 'interpolate'],
    )
    def test_every_way_lands_on_its_grid(self, position, column, stage, strategy, expected):
        """Same golden values whatever the position of the dates in their period."""
        data = _positioned_dataset(position)
        grid = data.index if stage == 'M' else data.index[data['q1'].notna()]
        features, _ways, _origins = _materialize(
            data, grid, stage, [column], {'m1': 'M', 'q1': 'Q', 'a1': 'Y'}, covariate_strategy=strategy
        )
        values = features[column]
        if column == 'a1':
            # Valeurs aux seules ancres annuelles, la grille étant mensuelle
            values = values.loc[data['a1'].dropna().index]
        np.testing.assert_allclose(values.to_numpy(), expected)

    @pytest.mark.parametrize('position', ['start', 'end'])
    def test_interpolation_fills_the_span_of_the_anchors(self, position):
        """Between the first and the last annual anchor, no month is left NaN."""
        data = _positioned_dataset(position)
        features, _ways, _origins = _materialize(
            data, data.index, 'M', ['a1'], {'a1': 'Y'}, covariate_strategy='interpolate'
        )
        anchors = data['a1'].dropna().index
        assert features['a1'].loc[anchors.min():anchors.max()].notna().all()
