"""Tests for helper functions in base module.

This module tests the utility functions used across the crossvals package.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest
import warnings

# Importation des fonctions auxiliaires
from tsforecast.crossvals.base import (
    _resolve_test_positions,
    _precompute_group_mappings,
    _precompute_multiindex_mappings,
    _verify_and_sort_data
)


class TestResolveTestPositions:
    """Test _resolve_test_positions function."""

    def test_none_input(self):
        """Test with None input returns None."""
        X = pd.Series(range(10))
        result = _resolve_test_positions(None, X)
        assert result is None

    def test_empty_list(self):
        """Test with empty list returns None."""
        X = pd.Series(range(10))
        result = _resolve_test_positions([], X)
        assert result is None

    def test_integer_positions(self):
        """Test with integer positions."""
        X = pd.Series(range(10))
        test_indices = [1, 3, 5]
        result = _resolve_test_positions(test_indices, X)
        np.testing.assert_array_equal(result, np.array([1, 3, 5]))

    def test_numpy_integer_positions(self):
        """Test with numpy integer positions."""
        X = pd.Series(range(10))
        test_indices = np.array([1, 3, 5])
        result = _resolve_test_positions(test_indices, X)
        np.testing.assert_array_equal(result, np.array([1, 3, 5]))

    def test_string_indices_timeseries(self):
        """Test with string indices for time series."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates.astype(str))
        test_indices = ['2020-01-02', '2020-01-05']
        result = _resolve_test_positions(test_indices, X)
        expected = np.array([1, 4])  # positions in the series
        np.testing.assert_array_equal(result, expected)

    def test_timestamp_indices_timeseries(self):
        """Test with timestamp indices for time series."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates)
        test_indices = [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-05')]
        result = _resolve_test_positions(test_indices, X)
        expected = np.array([1, 4])
        np.testing.assert_array_equal(result, expected)

    def test_tuple_indices_panel(self):
        """Test with tuple indices for panel data."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(10)}, index=idx)
        test_indices = [('A', dates[1]), ('B', dates[2])]
        result = _resolve_test_positions(test_indices, X)
        expected = np.array([1, 7])  # positions in the flattened index
        np.testing.assert_array_equal(result, expected)

    def test_invalid_indices_timeseries(self):
        """Test with invalid indices for time series."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates)
        test_indices = ['2020-02-01']  # not in the series
        
        with pytest.raises(ValueError, match="Time series index .* not found in data"):
            _resolve_test_positions(test_indices, X)

    def test_invalid_indices_panel(self):
        """Test with invalid indices for panel data."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(10)}, index=idx)
        test_indices = [('C', dates[1])]  # entity 'C' not in data
        
        with pytest.raises(ValueError, match="Panel index .* not found in data"):
            _resolve_test_positions(test_indices, X)

    def test_mixed_index_types_error(self):
        """Test that string indices work on panel data (all entities for that date)."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(10)}, index=idx)
        test_indices = ['2020-01-02']  # string for panel data (returns all entities for that date)
        
        # This should work and return positions for both entities at that date
        positions = _resolve_test_positions(test_indices, X)
        assert len(positions) == 2  # Should find both 'A' and 'B' at '2020-01-02'
        assert 1 in positions  # ('A', '2020-01-02')
        assert 6 in positions  # ('B', '2020-01-02')

    def test_no_pandas_index_error(self):
        """Test error when trying to use string indices without pandas index."""
        X = np.array(range(10))  # no pandas index
        test_indices = ['2020-01-02']
        
        with pytest.raises(ValueError, match="Cannot use string/date/tuple indices without pandas index"):
            _resolve_test_positions(test_indices, X)

    def test_scalar_input(self):
        """Test with scalar input."""
        X = pd.Series(range(10))
        result = _resolve_test_positions(5, X)
        np.testing.assert_array_equal(result, np.array([5]))


class TestPrecomputeGroupMappings:
    """Test _precompute_group_mappings function."""

    def test_simple_groups(self):
        """Test with simple group structure."""
        groups = np.array(['A', 'A', 'B', 'B', 'A'])
        result = _precompute_group_mappings(groups)
        
        expected = {
            'A': np.array([0, 1, 4]),
            'B': np.array([2, 3])
        }
        
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_single_group(self):
        """Test with single group."""
        groups = np.array(['A', 'A', 'A'])
        result = _precompute_group_mappings(groups)
        
        expected = {'A': np.array([0, 1, 2])}
        assert set(result.keys()) == set(expected.keys())
        np.testing.assert_array_equal(result['A'], expected['A'])

    def test_numeric_groups(self):
        """Test with numeric groups."""
        groups = np.array([1, 2, 1, 3, 2])
        result = _precompute_group_mappings(groups)
        
        expected = {
            1: np.array([0, 2]),
            2: np.array([1, 4]),
            3: np.array([3])
        }
        
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_empty_groups(self):
        """Test with empty groups array."""
        groups = np.array([])
        result = _precompute_group_mappings(groups)
        assert result == {}


class TestPrecomputeMultiindexMappings:
    """Test _precompute_multiindex_mappings function."""

    def test_multiindex_dataframe(self):
        """Test with MultiIndex DataFrame."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(6)}, index=idx)
        
        result = _precompute_multiindex_mappings(X)
        
        assert 'group_ranges' in result
        assert 'group_positions' in result
        assert 'level_0_values' in result
        assert 'level_1_values' in result
        
        # Check that we have mappings for both entities
        assert 'A' in result['group_ranges']
        assert 'B' in result['group_ranges']
        
        # Check level values
        np.testing.assert_array_equal(result['level_0_values'], ['A', 'A', 'A', 'B', 'B', 'B'])

    def test_single_index_dataframe(self):
        """Test with single-index DataFrame (should return empty mappings)."""
        X = pd.DataFrame({'value': range(5)})
        result = _precompute_multiindex_mappings(X)
        assert result == {}

    def test_series_with_multiindex(self):
        """Test with MultiIndex Series."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=2, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.Series(range(4), index=idx)
        
        result = _precompute_multiindex_mappings(X)
        
        assert 'group_ranges' in result
        assert 'group_positions' in result
        assert len(result['group_ranges']) == 2  # Two entities

    def test_no_pandas_structure(self):
        """Test with non-pandas structure."""
        X = np.array(range(5))
        result = _precompute_multiindex_mappings(X)
        assert result == {}


class TestVerifyAndSortData:
    """Test _verify_and_sort_data function."""

    def test_sorted_timeseries(self):
        """Test with already sorted time series."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        X = pd.Series(range(5), index=dates)
        
        X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups=None)
        
        # Should return original data since it's already sorted
        pd.testing.assert_series_equal(X_sorted, X)
        assert groups_sorted is None
        np.testing.assert_array_equal(sort_indices, np.arange(5))

    def test_unsorted_timeseries(self):
        """Test with unsorted time series."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        unsorted_dates = [dates[2], dates[0], dates[4], dates[1], dates[3]]
        X = pd.Series([2, 0, 4, 1, 3], index=unsorted_dates)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups=None)
            
            # Should issue warning about sorting
            assert len(w) == 1
            assert "not sorted by date" in str(w[0].message)
        
        # Check that data is now sorted
        assert X_sorted.index.is_monotonic_increasing
        assert groups_sorted is None

    def test_sorted_panel_data(self):
        """Test with sorted panel data."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(6)}, index=idx)
        groups = np.repeat(entities, 3)
        
        X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups=groups)
        
        # Should return original data since it's already sorted
        pd.testing.assert_frame_equal(X_sorted, X)
        np.testing.assert_array_equal(groups_sorted, groups)
        np.testing.assert_array_equal(sort_indices, np.arange(6))

    def test_unsorted_panel_data(self):
        """Test with unsorted panel data."""
        # Create unsorted panel data
        entities = ['B', 'A', 'B', 'A']
        dates = [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-01'), 
                pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
        idx = pd.MultiIndex.from_tuples(list(zip(entities, dates)), names=['entity', 'date'])
        X = pd.DataFrame({'value': [0, 1, 2, 3]}, index=idx)
        groups = np.array(['B', 'A', 'B', 'A'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups=groups)
            
            # Should issue warning about sorting
            assert len(w) == 1
            assert "not sorted by group then by date" in str(w[0].message)
        
        # Check that data is now properly sorted (A before B, dates ascending within each group)
        expected_order = [1, 3, 2, 0]  # A-2020-01-01, A-2020-01-02, B-2020-01-01, B-2020-01-02
        np.testing.assert_array_equal(sort_indices, expected_order)

    def test_no_pandas_index_timeseries(self):
        """Test with non-pandas data for time series."""
        X = np.array(range(5))
        
        X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups=None)
        
        # Should return original data
        np.testing.assert_array_equal(X_sorted, X)
        assert groups_sorted is None
        np.testing.assert_array_equal(sort_indices, np.arange(5))

    def test_invalid_panel_structure(self):
        """Test with invalid panel data structure."""
        X = pd.Series(range(5))  # Single index, not MultiIndex
        groups = np.array(['A', 'A', 'B', 'B', 'A'])
        
        with pytest.raises(ValueError, match="Panel data requires MultiIndex with at least 2 levels"):
            _verify_and_sort_data(X, groups=groups)

    def test_insufficient_multiindex_levels(self):
        """Test with insufficient MultiIndex levels."""
        X = pd.DataFrame({'value': range(5)}, index=pd.Index(range(5)))
        groups = np.array(['A', 'A', 'B', 'B', 'A'])
        
        with pytest.raises(ValueError, match="Panel data requires MultiIndex with at least 2 levels"):
            _verify_and_sort_data(X, groups=groups)