"""Tests for base cross-validation classes.

This module tests the core OutOfSampleSplit and InSampleSplit classes
that provide the foundation for all cross-validation functionality.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest
import warnings
# Importation des classes de base
from tsforecast.crossvals.base import OutOfSampleSplit, InSampleSplit


class TestOutOfSampleSplit:
    """Test OutOfSampleSplit class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = OutOfSampleSplit()
        assert splitter.n_splits == 5
        assert splitter.test_indices is None
        assert splitter.max_train_size is None
        assert splitter.test_size is None
        assert splitter.gap == 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        splitter = OutOfSampleSplit(
            n_splits=3,
            test_indices=[1, 2, 3],
            max_train_size=50,
            test_size=10,
            gap=2
        )
        assert splitter.n_splits == 3
        assert splitter.test_indices == [1, 2, 3]
        assert splitter.max_train_size == 50
        assert splitter.test_size == 10
        assert splitter.gap == 2

    def test_timeseries_basic_split(self):
        """Test basic time series split without groups."""
        X = pd.Series(range(20), index=pd.date_range('2020-01-01', periods=20, freq='D'))
        splitter = OutOfSampleSplit(n_splits=3, test_size=5)
        
        splits = list(splitter.split(X))
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            # Check that training comes before test
            assert max(train_idx) < min(test_idx)
            # Check test size
            assert len(test_idx) == 5
            # Check no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_timeseries_with_gap(self):
        """Test time series split with gap."""
        X = pd.Series(range(30), index=pd.date_range('2020-01-01', periods=30, freq='D'))
        splitter = OutOfSampleSplit(n_splits=2, test_size=5, gap=3)
        
        splits = list(splitter.split(X))
        
        for train_idx, test_idx in splits:
            # Check gap: there should be exactly 3 positions between last train and first test
            gap_positions = min(test_idx) - max(train_idx) - 1
            assert gap_positions == 3

    def test_timeseries_with_max_train_size(self):
        """Test time series split with max_train_size."""
        X = pd.Series(range(50), index=pd.date_range('2020-01-01', periods=50, freq='D'))
        splitter = OutOfSampleSplit(n_splits=2, test_size=5, max_train_size=20)
        
        splits = list(splitter.split(X))
        
        for train_idx, test_idx in splits:
            # Training size should not exceed max_train_size
            assert len(train_idx) <= 20

    def test_specific_test_indices_timeseries(self):
        """Test with specific test indices for time series."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.Series(range(20), index=dates)
        test_dates = ['2020-01-05', '2020-01-15']
        splitter = OutOfSampleSplit(test_indices=test_dates, test_size=3)
        
        splits = list(splitter.split(X))
        assert len(splits) == 2
        
        # Check that test indices correspond to specified dates
        for i, (train_idx, test_idx) in enumerate(splits):
            expected_start = X.index.get_loc(test_dates[i])
            assert min(test_idx) == expected_start
            assert len(test_idx) == 3

    def test_panel_data_basic_split(self):
        """Test basic panel data split."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        splitter = OutOfSampleSplit(n_splits=2, test_size=2)
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Check that we have test data for both entities
            test_groups = groups[test_idx]
            assert 'A' in test_groups
            assert 'B' in test_groups

    def test_panel_data_specific_test_indices(self):
        """Test panel data with specific test indices."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        test_dates = ['2020-01-05']
        splitter = OutOfSampleSplit(test_indices=test_dates, test_size=2)
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        
        # Should have test data for both entities at the specified date
        assert len(test_idx) == 4  # 2 periods × 2 entities

    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = pd.Series([], dtype=float)
        splitter = OutOfSampleSplit(n_splits=2, test_size=1)
        
        with pytest.raises(ValueError):
            list(splitter.split(X))

    def test_single_observation(self):
        """Test with single observation."""
        X = pd.Series([1])
        splitter = OutOfSampleSplit(n_splits=2, test_size=1)
        
        # Should raise error for impossible configuration (1 sample with n_splits=2)
        with pytest.raises(ValueError, match="Cannot have number of folds"):
            list(splitter.split(X))

    def test_invalid_n_splits(self):
        """Test with invalid n_splits parameter."""
        X = pd.Series(range(5))
        
        # Test with too many splits
        splitter = OutOfSampleSplit(n_splits=10, test_size=1)
        with pytest.raises(ValueError, match="Cannot have number of folds"):
            list(splitter.split(X))

    def test_invalid_test_size(self):
        """Test with invalid test_size parameter."""
        X = pd.Series(range(10))
        
        # Test with test_size larger than data
        splitter = OutOfSampleSplit(n_splits=2, test_size=15)
        
        # Should raise error for impossible configuration
        with pytest.raises(ValueError, match="Too many splits"):
            list(splitter.split(X))

    def test_xy_index_mismatch(self):
        """Test with mismatched X and y indices."""
        X = pd.Series(range(10), index=pd.date_range('2020-01-01', periods=10, freq='D'))
        y = pd.Series(range(10), index=pd.date_range('2020-01-02', periods=10, freq='D'))
        
        splitter = OutOfSampleSplit(n_splits=2, test_size=2)
        
        with pytest.raises(ValueError, match="'X' and 'y' should have the same indexes"):
            list(splitter.split(X, y))

    def test_unsorted_data_warning(self):
        """Test warning for unsorted data."""
        # Create unsorted time series
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        unsorted_dates = [dates[2], dates[0], dates[4], dates[1], dates[3]]
        X = pd.Series([2, 0, 4, 1, 3], index=unsorted_dates)
        
        splitter = OutOfSampleSplit(n_splits=2, test_size=1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(splitter.split(X))
            
            # Should issue warning about sorting
            assert len(w) >= 1
            assert any("not sorted by date" in str(warning.message) for warning in w)

    def test_gap_larger_than_data(self):
        """Test with gap larger than available training data."""
        X = pd.Series(range(10))
        splitter = OutOfSampleSplit(n_splits=2, test_size=2, gap=15)
        
        splits = list(splitter.split(X))
        
        # With a large gap, some or all splits might result in empty training sets
        for train_idx, test_idx in splits:
            # Training set should be empty or very small due to large gap
            assert len(train_idx) == 0 or len(train_idx) < 3

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X = np.arange(20)
        splitter = OutOfSampleSplit(n_splits=2, test_size=3)
        
        splits = list(splitter.split(X))
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(test_idx) == 3

    def test_test_indices_beyond_data_length(self):
        """Test with test_indices beyond data length."""
        X = pd.Series(range(10))
        splitter = OutOfSampleSplit(test_indices=[15, 20], test_size=1)
        
        with pytest.raises(ValueError, match="test_indices contains indices beyond data length"):
            list(splitter.split(X))

    def test_negative_test_indices(self):
        """Test with negative test indices."""
        X = pd.Series(range(10))
        splitter = OutOfSampleSplit(test_indices=[-1, -2], test_size=1)
        
        with pytest.raises(ValueError, match="test_indices contains negative indices"):
            list(splitter.split(X))


class TestInSampleSplit:
    """Test InSampleSplit class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = InSampleSplit()
        assert splitter.n_splits == 5
        assert splitter.test_indices is None
        assert splitter.max_train_size is None
        assert splitter.test_size is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        splitter = InSampleSplit(
            n_splits=3,
            test_indices=[1, 2, 3],
            max_train_size=50,
            test_size=10
        )
        assert splitter.n_splits == 3
        assert splitter.test_indices == [1, 2, 3]
        assert splitter.max_train_size == 50
        assert splitter.test_size == 10

    def test_timeseries_basic_split(self):
        """Test basic in-sample time series split."""
        X = pd.Series(range(20), index=pd.date_range('2020-01-01', periods=20, freq='D'))
        splitter = InSampleSplit(test_size=5)
        
        splits = list(splitter.split(X))
        assert len(splits) == 1  # In-sample typically has one split
        
        train_idx, test_idx = splits[0]
        
        # Training should include test period (in-sample characteristic)
        assert max(test_idx) <= max(train_idx)
        # Test indices should be subset of training indices
        assert np.all(np.isin(test_idx, train_idx))

    def test_timeseries_with_max_train_size(self):
        """Test in-sample split with max_train_size."""
        X = pd.Series(range(50), index=pd.date_range('2020-01-01', periods=50, freq='D'))
        splitter = InSampleSplit(test_size=5, max_train_size=20)
        
        splits = list(splitter.split(X))
        train_idx, test_idx = splits[0]
        
        # Training size should respect max_train_size but include test period
        assert len(train_idx) <= 20
        # Test period should still be included
        assert np.all(np.isin(test_idx, train_idx))

    def test_specific_test_indices_single(self):
        """Test with single specific test index."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.Series(range(20), index=dates)
        test_dates = ['2020-01-10']
        splitter = InSampleSplit(test_indices=test_dates, test_size=3)
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        # Check that test starts at specified date
        expected_start = X.index.get_loc(test_dates[0])
        assert min(test_idx) == expected_start
        assert len(test_idx) == 3
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_specific_test_indices_multiple(self):
        """Test with multiple specific test indices (should include all)."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.Series(range(20), index=dates)
        test_dates = ['2020-01-05', '2020-01-10', '2020-01-15']
        splitter = InSampleSplit(test_indices=test_dates, test_size=2)  # test_size ignored
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        # Should include all specified test indices
        assert len(test_idx) == 3  # All three test dates
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_panel_data_basic_split(self):
        """Test basic panel data in-sample split."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        splitter = InSampleSplit(test_size=2)
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        
        # Training should include test period for panel data
        assert np.all(np.isin(test_idx, train_idx))

    def test_panel_data_specific_test_indices_single(self):
        """Test panel data with single specific test index."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        test_dates = ['2020-01-05']
        splitter = InSampleSplit(test_indices=test_dates, test_size=2)
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        
        # Should have test data for both entities at the specified date
        assert len(test_idx) == 4  # 2 periods × 2 entities
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_panel_data_specific_test_indices_multiple(self):
        """Test panel data with multiple specific test indices."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        test_dates = ['2020-01-03', '2020-01-07']
        splitter = InSampleSplit(test_indices=test_dates, test_size=1)  # test_size ignored
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        
        # Should include all specified test dates for both entities
        assert len(test_idx) == 4  # 2 dates × 2 entities
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = pd.Series([], dtype=float)
        splitter = InSampleSplit(test_size=1)
        
        with pytest.raises(ValueError):
            list(splitter.split(X))

    def test_single_observation(self):
        """Test with single observation."""
        X = pd.Series([1])
        splitter = InSampleSplit(test_size=1)
        
        splits = list(splitter.split(X))
        train_idx, test_idx = splits[0]
        
        # Should have single observation in both train and test
        assert len(train_idx) == 1
        assert len(test_idx) == 1
        assert train_idx[0] == test_idx[0]

    def test_xy_index_mismatch(self):
        """Test with mismatched X and y indices."""
        X = pd.Series(range(10), index=pd.date_range('2020-01-01', periods=10, freq='D'))
        y = pd.Series(range(10), index=pd.date_range('2020-01-02', periods=10, freq='D'))
        
        splitter = InSampleSplit(test_size=2)
        
        with pytest.raises(ValueError, match="'X' and 'y' should have the same indexes"):
            list(splitter.split(X, y))

    def test_unsorted_data_warning(self):
        """Test warning for unsorted data."""
        # Create unsorted time series
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        unsorted_dates = [dates[2], dates[0], dates[4], dates[1], dates[3]]
        X = pd.Series([2, 0, 4, 1, 3], index=unsorted_dates)
        
        splitter = InSampleSplit(test_size=1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(splitter.split(X))
            
            # Should issue warning about sorting
            assert len(w) == 1
            assert "not sorted by date" in str(w[0].message)

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X = np.arange(20)
        splitter = InSampleSplit(test_size=3)
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(test_idx) == 3
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_test_indices_beyond_data_length(self):
        """Test with test_indices beyond data length."""
        X = pd.Series(range(10))
        splitter = InSampleSplit(test_indices=[15, 20], test_size=1)
        
        with pytest.raises(ValueError, match="test_indices contains indices beyond data length"):
            list(splitter.split(X))

    def test_negative_test_indices(self):
        """Test with negative test indices."""
        X = pd.Series(range(10))
        splitter = InSampleSplit(test_indices=[-1, -2], test_size=1)
        
        with pytest.raises(ValueError, match="test_indices contains negative indices"):
            list(splitter.split(X))

    def test_max_train_size_smaller_than_test(self):
        """Test max_train_size smaller than test size."""
        X = pd.Series(range(20))
        splitter = InSampleSplit(test_size=10, max_train_size=5)
        
        splits = list(splitter.split(X))
        train_idx, test_idx = splits[0]
        
        # Training should be at least as large as test (since test is included)
        assert len(train_idx) >= len(test_idx)
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))