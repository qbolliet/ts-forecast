"""Unit tests of ``tsforecast.adapters.hierarchicalforecast._temporal``.

``parse_temporal_hierarchy`` (ordering, nested and grouped levels, malformed
mappings), ``temporal_structure`` (summing matrix of one period), ``cross_structure``
(Kronecker product, aggregates first, bottom identity, cross level names),
``TemporalCalendar`` (calendar periods of start- and end-labelled grids, hourly
and weekly grids, ``origin``, dating of the aggregates, grids that cannot carry
the hierarchy, date resolution), ``read_temporal_index`` and
``detect_fine_frequency``.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import DATETIME_UNITS
from tsforecast.adapters._base import TemporalLayout
from tsforecast.adapters.hierarchicalforecast._structure import tree_structure
from tsforecast.adapters.hierarchicalforecast._temporal import (
    TemporalCalendar,
    cross_structure,
    detect_fine_frequency,
    parse_temporal_hierarchy,
    read_temporal_index,
    temporal_structure,
)

# Hiérarchie temporelle mensuelle de référence
MONTHLY = {'year': 12, 'quarter': 3, 'month': 1}


class TestParseTemporalHierarchy:
    """Reading of the ``{name: number of fine steps}`` mapping."""

    def test_levels_sorted_most_aggregated_first(self):
        spec = parse_temporal_hierarchy({'month': 1, 'year': 12, 'quarter': 3})
        assert spec.levels == ('year', 'quarter', 'month')
        assert spec.steps == (12, 3, 1)
        assert (spec.period_steps, spec.finest) == (12, 'month')
        assert spec.counts().tolist() == [1, 4, 12]
        assert spec.offsets().tolist() == [0, 1, 5]

    def test_grouped_levels_accepted(self):
        spec = parse_temporal_hierarchy({'year': 12, 'four_months': 4, 'quarter': 3, 'month': 1})
        assert spec.levels == ('year', 'four_months', 'quarter', 'month')

    def test_numpy_integers_accepted(self):
        assert parse_temporal_hierarchy({'year': np.int64(4), 'quarter': 1}).steps == (4, 1)

    @pytest.mark.parametrize(
        ('hierarchy', 'message'),
        [
            ({}, 'non-empty'),
            ([('year', 12)], 'non-empty'),
            ({'year': 12, '': 1}, 'non-empty strings'),
            ({'year': 12, 3: 1}, 'non-empty strings'),
            ({'year': 0, 'month': 1}, 'positive integer'),
            ({'year': 1.5, 'month': 1}, 'positive integer'),
            ({'year': True, 'month': 1}, 'positive integer'),
            ({'year': 12, 'quarter': 3}, 'covering 1 step'),
            ({'year': 12, 'annual': 12, 'month': 1}, 'same number of steps'),
            ({'year': 12, 'five': 5, 'month': 1}, 'divide'),
        ],
    )
    def test_malformed_hierarchy(self, hierarchy, message):
        with pytest.raises(ValueError, match=message):
            parse_temporal_hierarchy(hierarchy)


class TestTemporalStructure:
    """Summing matrix of one period of the most aggregated level."""

    def test_monthly_structure(self):
        structure = temporal_structure(parse_temporal_hierarchy(MONTHLY))
        assert structure.n_nodes == 17 and structure.n_bottom == 12
        assert structure.node_keys[:6] == ('year-1', 'quarter-1', 'quarter-2', 'quarter-3', 'quarter-4', 'month-1')
        np.testing.assert_array_equal(structure.summing[0], np.ones(12))
        np.testing.assert_array_equal(structure.summing[2], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(structure.summing[5:], np.eye(12))
        assert structure.levels == ('year', 'quarter', 'month')
        assert structure.bottom_is_finest_level()

    def test_tags_name_levels(self):
        structure = temporal_structure(parse_temporal_hierarchy({'year': 4, 'semester': 2, 'quarter': 1}))
        assert {level: len(ids) for level, ids in structure.tags().items()} == {'year': 1, 'semester': 2, 'quarter': 4}


class TestCrossStructure:
    """Kronecker product of a cross-sectional and a temporal structure."""

    @pytest.fixture
    def cross(self):
        temporal = temporal_structure(parse_temporal_hierarchy({'year': 2, 'half': 1}))
        return cross_structure(tree_structure({'a': ['b', 'c'], 'c': ['d', 'e']}), temporal)

    def test_shape_levels_and_bottom_identity(self, cross):
        structure = cross.structure
        assert structure.summing.shape == (15, 6)
        np.testing.assert_array_equal(structure.summing[-6:], np.eye(6))
        assert structure.levels[0] == 'depth_0//year' and structure.levels[-1] == 'depth_2//half'
        assert len(structure.levels) == 6

    def test_rows_are_kronecker_rows(self, cross):
        kron = np.kron(cross.cross_section.summing, temporal_structure(
            parse_temporal_hierarchy({'year': 2, 'half': 1})).summing)
        for pair, position in enumerate(cross.positions):
            np.testing.assert_array_equal(cross.structure.summing[position], kron[pair])
        assert sorted(cross.positions.tolist()) == list(range(15))

    def test_node_keys_and_level_of_each_pair(self, cross):
        position = cross.positions[1 * 3 + 0]
        assert cross.structure.node_keys[position] == ('c', 'year-1')
        assert cross.structure.levels[cross.structure.node_levels[position]] == 'depth_1//year'

    def test_aggregates_ordered_by_cross_level(self, cross):
        aggregates = cross.structure.node_levels[:-6]
        assert np.all(np.diff(aggregates) >= 0)

    def test_positions_are_read_only(self, cross):
        with pytest.raises(ValueError):
            cross.positions[0] = 1


class TestTemporalCalendar:
    """Calendar periods, nodes and dating of the aggregates."""

    @pytest.mark.parametrize(
        ('freq', 'hierarchy', 'period_freq', 'end_labelled'),
        [
            ('MS', MONTHLY, '12MS', False),
            ('ME', MONTHLY, '12ME', True),
            ('QS-JAN', {'year': 4, 'quarter': 1}, '4QS-JAN', False),
            ('QE-DEC', {'year': 4, 'quarter': 1}, '4QE-DEC', True),
            ('h', {'day': 24, 'hour': 1}, '24h', False),
            ('W-SUN', {'four_weeks': 4, 'week': 1}, '4W-SUN', True),
        ],
    )
    def test_period_frequency_and_labelling(self, freq, hierarchy, period_freq, end_labelled):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(hierarchy), freq)
        assert (calendar.period_freq, calendar.end_labelled) == (period_freq, end_labelled)

    def test_start_labelled_grid(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'MS')
        dates = pd.DatetimeIndex(['2024-01-01', '2024-04-01', '2024-11-01', '2025-01-01'])
        placement = calendar.locate(np.array([0, 1, 2, 1]), dates)
        assert placement.period_start.strftime('%Y-%m-%d').tolist() == ['2024-01-01'] * 3 + ['2025-01-01']
        # Nœuds : année, 2e trimestre, 11e mois, 1er trimestre
        assert placement.node.tolist() == [0, 2, 15, 1]
        assert placement.last.strftime('%Y-%m-%d').tolist() == ['2024-12-01', '2024-06-01', '2024-11-01', '2025-03-01']

    def test_end_labelled_grid(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'ME')
        dates = pd.DatetimeIndex(['2024-12-31', '2024-06-30', '2024-01-31'])
        placement = calendar.locate(np.array([0, 1, 2]), dates)
        assert placement.period_start.strftime('%Y-%m-%d').tolist() == ['2024-01-01'] * 3
        assert placement.node.tolist() == [0, 2, 5]
        assert placement.last.equals(dates)

    def test_hourly_grid_has_daily_periods(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy({'day': 24, 'hour': 1}), 'h')
        placement = calendar.locate(np.array([1, 0]), pd.DatetimeIndex(['2024-03-02 05:00', '2024-03-02 00:00']))
        assert placement.period_start.strftime('%Y-%m-%d %H').tolist() == ['2024-03-02 00'] * 2
        assert placement.node.tolist() == [1 + 5, 0]

    def test_origin_shifts_the_periods(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'MS', origin='2023-10-01')
        placement = calendar.locate(np.array([2, 0]), pd.DatetimeIndex(['2024-09-01', '2024-10-01']))
        assert placement.period_start.strftime('%Y-%m-%d').tolist() == ['2023-10-01', '2024-10-01']
        assert placement.node.tolist() == [5 + 11, 0]

    def test_daily_weeks_from_a_monday_origin(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy({'week': 7, 'day': 1}), 'D', origin='2024-01-01')
        placement = calendar.locate(np.array([0]), pd.DatetimeIndex(['2024-01-08']))
        assert placement.period_start[0] == pd.Timestamp('2024-01-08')
        assert placement.period_start[0].day_name() == 'Monday'

    def test_invalid_origin(self):
        with pytest.raises(ValueError, match='temporal_origin must be a date'):
            TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'MS', origin='not a date')

    @pytest.mark.parametrize(
        ('freq', 'date', 'message'),
        [
            ('MS', '2024-02-01', 'first'),
            ('ME', '2024-01-31', 'last'),
        ],
    )
    def test_misdated_aggregate(self, freq, date, message):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), freq)
        with pytest.raises(ValueError, match=f"level 'quarter' is dated .* not the {message}"):
            calendar.locate(np.array([1]), pd.DatetimeIndex([date]))

    def test_date_off_the_grid(self):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'MS')
        with pytest.raises(ValueError, match='not on the'):
            calendar.locate(np.array([2]), pd.DatetimeIndex(['2024-01-15']))

    @pytest.mark.parametrize(
        ('freq', 'hierarchy'),
        [('B', {'week': 5, 'day': 1}), ('2MS', {'year': 6, 'bimester': 1})],
    )
    def test_grid_that_does_not_tile_the_periods(self, freq, hierarchy):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(hierarchy), freq)
        # Jours ouvrés : une semaine de 5 jours ouvrés ne s'aligne pas sur l'époque ;
        # bimestres déphasés (février, avril...) : dates hors de la grille des années
        dates = pd.DatetimeIndex(['2024-05-01']) if freq == 'B' else pd.DatetimeIndex(['2024-02-01'])
        with pytest.raises(ValueError, match='cannot carry|not on the'):
            calendar.locate(np.array([1]), dates)

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_resolution_does_not_change_the_placement(self, unit):
        calendar = TemporalCalendar.build(parse_temporal_hierarchy(MONTHLY), 'MS')
        dates = pd.DatetimeIndex(['2024-04-01', '2025-12-01']).as_unit(unit)
        placement = calendar.locate(np.array([1, 2]), dates)
        assert placement.node.tolist() == [2, 16]
        assert placement.period_start.equals(pd.DatetimeIndex(['2024-01-01', '2025-01-01']))


class TestTemporalIndex:
    """Reading of the temporal index level and detection of the fine grid."""

    @staticmethod
    def _layout(levels, dates, names=('entity', 'level', 'date'), entities=None):
        entities = ['A'] * len(levels) if entities is None else entities
        index = pd.MultiIndex.from_arrays([entities, levels, pd.DatetimeIndex(dates)], names=list(names))
        return TemporalLayout.from_index(index, detect_freq=False)

    def test_level_codes_and_default_position(self):
        layout = self._layout(['month', 'month', 'year'], ['2024-01-01', '2024-02-01', '2024-01-01'])
        temporal = read_temporal_index(layout, parse_temporal_hierarchy({'year': 12, 'month': 1}))
        assert temporal.position == 1
        assert temporal.row_levels.tolist() == [1, 1, 0]

    def test_level_by_name_and_position(self):
        spec = parse_temporal_hierarchy({'year': 12, 'month': 1})
        # Niveau temporel en tête de l'index, désigné par son nom ou sa position
        index = pd.MultiIndex.from_arrays(
            [['month', 'year'], ['A', 'A'], pd.DatetimeIndex(['2024-01-01', '2024-01-01'])],
            names=['level', 'entity', 'date'],
        )
        layout = TemporalLayout.from_index(index, detect_freq=False)
        assert read_temporal_index(layout, spec, 'level').row_levels.tolist() == [1, 0]
        assert read_temporal_index(layout, spec, 0).position == 0

    def test_unknown_label(self):
        layout = self._layout(['month', 'week'], ['2024-01-01', '2024-01-01'])
        with pytest.raises(ValueError, match="'week' of the temporal index level 'level'"):
            read_temporal_index(layout, parse_temporal_hierarchy({'year': 12, 'month': 1}))

    def test_no_level_before_the_date(self):
        layout = TemporalLayout.from_index(pd.date_range('2024-01-01', periods=2, freq='MS'))
        with pytest.raises(ValueError, match='before the date level'):
            read_temporal_index(layout, parse_temporal_hierarchy({'year': 12, 'month': 1}))

    def test_fine_frequency_read_on_the_finest_rows(self):
        months = pd.date_range('2024-01-31', periods=12, freq='ME')
        layout = self._layout(['month'] * 12 + ['year'], months.append(months[-1:]), entities=['A'] * 13)
        spec = parse_temporal_hierarchy({'year': 12, 'month': 1})
        temporal = read_temporal_index(layout, spec)
        assert detect_fine_frequency(layout, temporal.row_levels, spec) == 'ME'

    def test_undetectable_fine_frequency(self):
        layout = self._layout(['month', 'year'], ['2024-01-01', '2024-01-01'])
        spec = parse_temporal_hierarchy({'year': 12, 'month': 1})
        temporal = read_temporal_index(layout, spec)
        with pytest.raises(ValueError, match="grid of the finest temporal level 'month' cannot be detected"):
            detect_fine_frequency(layout, temporal.row_levels, spec)
