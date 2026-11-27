"""Unit tests of ``tsforecast.adapters.foundation._series``.

Grid positions of the rows of each entity (holes, unordered rows, multiplied
frequencies), contexts laid on the grids (holes as ``NaN``, most recent steps
kept), windows of the rows to predict, canonical entity rank (comparable and
incomparable labels), validation of quantile levels, native levels around the
requested ones and linear interpolation.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.adapters._base import TemporalLayout
from tsforecast.adapters.foundation._series import (
    bracketing_levels,
    check_quantile_levels,
    entity_grid_positions,
    entity_rank,
    future_windows,
    interpolate_quantiles,
    place_on_grids,
)
from tsforecast.adapters.foundation.protocol import BackendCapabilities


class TestGrids:
    """Rows laid on the grid of their entity."""

    def test_positions_with_holes_and_unordered_rows(self):
        dates = pd.to_datetime(['2024-07-01', '2024-01-01', '2024-05-01', '2024-03-01'])
        index = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'B'], dates])
        layout = TemporalLayout.from_index(index, freq='2MS')
        positions, lengths = entity_grid_positions(layout, '2MS')
        assert positions.tolist() == [3, 0, 1, 0] and lengths.tolist() == [4, 2]

    def test_place_on_grids(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])
        grids = place_on_grids(values, np.array([0, 0, 1]), np.array([2, 0, 0]), np.array([3, 1]))
        np.testing.assert_array_equal(grids[0], [[2.0, np.nan, 1.0], [20.0, np.nan, 10.0]])
        assert not grids[0].flags.writeable
        kept = place_on_grids(values, np.array([0, 0, 1]), np.array([2, 0, 0]), np.array([3, 1]), keep=2)
        np.testing.assert_array_equal(kept[0], [[np.nan, 1.0], [np.nan, 10.0]])
        np.testing.assert_array_equal(kept[1], [[5.0], [50.0]])

    def test_future_windows(self):
        windows = future_windows(np.array([[1.0], [2.0], [3.0]]), np.array([2, 0, 2]), np.array([2, 1, 4]), 3)
        np.testing.assert_array_equal(windows[0], [[2.0]])
        assert windows[1] is None
        np.testing.assert_array_equal(windows[2], [[np.nan, 1.0, np.nan, 3.0]])


class TestEntityRank:
    """Order of the entities by label."""

    def test_comparable_and_incomparable_labels(self):
        assert entity_rank([('b', 2), ('a', 9), ('b', 1)]).tolist() == [2, 0, 1]
        assert entity_rank([(1,), ('a',)]).tolist() == [0, 1]
        assert entity_rank([()]).tolist() == [0]


class TestQuantileLevels:
    """Validation and interpolation of quantile levels."""

    def test_range(self):
        native = BackendCapabilities(quantile_levels=(0.1, 0.5, 0.9))
        np.testing.assert_array_equal(check_quantile_levels((0.9, 0.1), native, 'levels'), [0.9, 0.1])
        with pytest.raises(ValueError, match=r'\[0.95\] lie outside'):
            check_quantile_levels([0.5, 0.95], native, 'levels')
        assert check_quantile_levels(0.05, BackendCapabilities(), 'levels').tolist() == [0.05]

    @pytest.mark.parametrize('levels', [[], [np.nan], [-0.1], [0.3, 0.3], [[0.1, 'x']]])
    def test_invalid(self, levels):
        with pytest.raises(ValueError):
            check_quantile_levels(levels, BackendCapabilities(), 'levels')

    def test_bracketing_levels(self):
        native = (0.1, 0.2, 0.5, 0.9)
        assert bracketing_levels(np.array([0.2]), native) == (0.2,)
        assert bracketing_levels(np.array([0.3, 0.6, 0.9]), native) == (0.2, 0.5, 0.9)

    def test_interpolation(self):
        values = np.array([[[0.0, 4.0, 8.0]], [[1.0, 1.0, 2.0]]])
        result = interpolate_quantiles(values, (0.25, 0.5, 0.75), (0.75, 0.375, 0.25))
        np.testing.assert_array_equal(result, [[[8.0, 2.0, 0.0]], [[2.0, 1.0, 1.0]]])
