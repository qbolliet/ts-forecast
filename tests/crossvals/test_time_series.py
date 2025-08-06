"""Tests for time series cross-validation classes.

This module tests the TSOutOfSampleSplit and TSInSampleSplit classes
specialized for time series data.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from tsforecast.crossvals.time_series import TSOutOfSampleSplit, TSInSampleSplit


class TestTSOutOfSampleSplit:
    """Test TSOutOfSampleSplit class."""

    def test_init_inherits_from_base(self):
        """Test that class inherits from OutOfSampleSplit."""
        from tsforecast.crossvals.base import OutOfSampleSplit
        splitter = TSOutOfSampleSplit()
        assert isinstance(splitter, OutOfSampleSplit)

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = TSOutOfSampleSplit()
        assert splitter.n_splits == 5
        assert splitter.test_indices is None
        assert splitter.max_train_size is None
        assert splitter.test_size is None
        assert splitter.gap == 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        splitter = TSOutOfSampleSplit(
            n_splits=3,
            test_indices=['2020-01-05'],
            max_train_size=50,
            test_size=10,
            gap=2
        )
        assert splitter.n_splits == 3
        assert splitter.test_indices == ['2020-01-05']
        assert splitter.max_train_size == 50
        assert splitter.test_size == 10
        assert splitter.gap == 2

    def test_basic_timeseries_split(self):
        """Test basic time series split."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=5, gap=1)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            # Check that training comes before test
            assert max(train_idx) < min(test_idx)
            # Check test size
            assert len(test_idx) == 5
            # Check gap
            gap_positions = min(test_idx) - max(train_idx) - 1
            assert gap_positions == 1

    def test_timeseries_with_datetime_index(self):
        """Test with proper datetime index."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.Series(np.random.randn(20), index=dates)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Verify indices are valid
            assert all(0 <= idx < len(X) for idx in train_idx)
            assert all(0 <= idx < len(X) for idx in test_idx)
            # Verify temporal ordering
            assert max(train_idx) < min(test_idx)

    def test_timeseries_with_string_index(self):
        """Test with string-based index."""
        X = pd.Series(range(10), index=[f'period_{i}' for i in range(10)])
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2)
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            assert len(test_idx) == 2
            assert max(train_idx) < min(test_idx)

    def test_timeseries_with_range_index(self):
        """Test with simple range index."""
        X = pd.DataFrame({'value': range(15)})
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(test_idx) == 3
            assert max(train_idx) < min(test_idx)

    def test_specific_test_dates(self):
        """Test with specific test dates."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        test_dates = ['2020-01-10', '2020-01-20']
        splitter = TSOutOfSampleSplit(test_indices=test_dates, test_size=3, gap=2)
        
        splits = list(splitter.split(X))
        assert len(splits) == 2
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check that test starts at the right date
            expected_start = X.index.get_loc(test_dates[i])
            assert min(test_idx) == expected_start
            assert len(test_idx) == 3
            # Check gap
            gap_positions = min(test_idx) - max(train_idx) - 1
            assert gap_positions == 2

    def test_specific_test_positions(self):
        """Test with specific numeric test positions."""
        X = pd.Series(range(20))
        
        test_positions = [5, 15]
        splitter = TSOutOfSampleSplit(test_indices=test_positions, test_size=2)
        
        splits = list(splitter.split(X))
        assert len(splits) == 2
        
        for i, (train_idx, test_idx) in enumerate(splits):
            assert min(test_idx) == test_positions[i]
            assert len(test_idx) == 2

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X = np.arange(20)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(test_idx) == 3
            assert max(train_idx) < min(test_idx)

    def test_groups_parameter_warning(self):
        """Test that groups parameter issues warning and is ignored."""
        X = pd.Series(range(10))
        groups = np.array(['A'] * 5 + ['B'] * 5)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splits = list(splitter.split(X, groups=groups))
            
            # Should issue warning about ignoring groups
            assert len(w) == 1
            assert "groups parameter is ignored" in str(w[0].message)
            assert "Time series splits treat data as a single temporal sequence" in str(w[0].message)
        
        # Should still work normally
        assert len(splits) == 2  # With n_splits=2
        train_idx, test_idx = splits[0]
        assert len(test_idx) == 2

    def test_rolling_window_behavior(self):
        """Test rolling window validation behavior."""
        X = pd.Series(range(50))
        
        splitter = TSOutOfSampleSplit(n_splits=5, test_size=5, max_train_size=15)
        splits = list(splitter.split(X))
        
        assert len(splits) == 5
        
        for train_idx, test_idx in splits:
            # Training window should not exceed max_train_size
            assert len(train_idx) <= 15
            # Test size should be consistent
            assert len(test_idx) == 5
            # Temporal ordering maintained
            assert max(train_idx) < min(test_idx)

    def test_expanding_window_behavior(self):
        """Test expanding window validation behavior."""
        X = pd.Series(range(25))
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=5)  # no max_train_size
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        # Training windows should expand (get larger) in expanding window mode
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_insufficient_data_for_splits(self):
        """Test with insufficient data for the requested number of splits."""
        X = pd.Series(range(8))
        
        splitter = TSOutOfSampleSplit(n_splits=5, test_size=3)
        
        with pytest.raises(ValueError, match="Too many splits"):
            list(splitter.split(X))

    def test_test_size_larger_than_data(self):
        """Test with test_size larger than available data."""
        X = pd.Series(range(5))
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=10)
        
        # Should raise error for impossible configuration
        with pytest.raises(ValueError, match="Too many splits"):
            list(splitter.split(X))

    def test_gap_leaves_no_training_data(self):
        """Test with gap that leaves no training data."""
        X = pd.Series(range(10))
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=3, gap=15)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        # Should result in empty training set
        assert len(train_idx) == 0
        assert len(test_idx) == 3

    def test_single_observation(self):
        """Test with single observation."""
        X = pd.Series([42])
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=1)
        
        # Should raise error for impossible configuration (1 sample with n_splits=2)
        with pytest.raises(ValueError, match="Cannot have number of folds"):
            list(splitter.split(X))

    def test_empty_series(self):
        """Test with empty series."""
        X = pd.Series([], dtype=float)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=1)
        
        with pytest.raises(ValueError):
            list(splitter.split(X))

    def test_invalid_test_dates(self):
        """Test with invalid test dates."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates)
        
        test_dates = ['2020-02-01']  # Not in the series
        splitter = TSOutOfSampleSplit(test_indices=test_dates, test_size=1)
        
        with pytest.raises(ValueError, match="not found in data"):
            list(splitter.split(X))

    def test_mixed_index_types(self):
        """Test behavior with different index types."""
        # Test datetime index
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X_dt = pd.Series(range(10), index=dates)
        
        # Test integer index  
        X_int = pd.Series(range(10))
        
        # Test string index
        X_str = pd.Series(range(10), index=[f'item_{i}' for i in range(10)])
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2)
        
        for X in [X_dt, X_int, X_str]:
            splits = list(splitter.split(X))
            assert len(splits) == 2
            
            for train_idx, test_idx in splits:
                assert len(test_idx) == 2
                assert max(train_idx) < min(test_idx)

    def test_y_parameter_handling(self):
        """Test proper handling of y parameter."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.DataFrame({'feature': range(20)}, index=dates)
        y = pd.Series(range(20, 40), index=dates)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=3)
        splits = list(splitter.split(X, y))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Verify we can use indices to slice both X and y
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)


class TestTSInSampleSplit:
    """Test TSInSampleSplit class."""

    def test_init_inherits_from_base(self):
        """Test that class inherits from InSampleSplit."""
        from tsforecast.crossvals.base import InSampleSplit
        splitter = TSInSampleSplit()
        assert isinstance(splitter, InSampleSplit)

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = TSInSampleSplit()
        assert splitter.n_splits == 5
        assert splitter.test_indices is None
        assert splitter.max_train_size is None
        assert splitter.test_size is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        splitter = TSInSampleSplit(
            n_splits=3,
            test_indices=['2020-01-15'],
            max_train_size=50,
            test_size=10
        )
        assert splitter.n_splits == 3
        assert splitter.test_indices == ['2020-01-15']
        assert splitter.max_train_size == 50
        assert splitter.test_size == 10

    def test_basic_insample_split(self):
        """Test basic in-sample time series split."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        splitter = TSInSampleSplit(test_size=10)
        splits = list(splitter.split(X))
        
        assert len(splits) == 1  # In-sample typically produces one split
        
        train_idx, test_idx = splits[0]
        
        # Key characteristic: training includes test period
        assert max(test_idx) <= max(train_idx)
        assert np.all(np.isin(test_idx, train_idx))

    def test_insample_with_datetime_index(self):
        """Test in-sample split with datetime index."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.Series(np.random.randn(20), index=dates)
        
        splitter = TSInSampleSplit(test_size=5)
        splits = list(splitter.split(X))
        
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        assert len(test_idx) == 5
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_insample_specific_test_date(self):
        """Test in-sample split with specific test date."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        test_date = ['2020-01-15']
        splitter = TSInSampleSplit(test_indices=test_date, test_size=5)
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        
        # Check that test starts at the right date
        expected_start = X.index.get_loc(test_date[0])
        assert min(test_idx) == expected_start
        assert len(test_idx) == 5
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_insample_multiple_test_dates(self):
        """Test in-sample split with multiple test dates (should include all)."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        test_dates = ['2020-01-10', '2020-01-15', '2020-01-20']
        splitter = TSInSampleSplit(test_indices=test_dates, test_size=2)  # test_size ignored
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        
        # Should include all specified test dates
        assert len(test_idx) == 3
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_insample_with_max_train_size(self):
        """Test in-sample split with max_train_size."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        X = pd.DataFrame({'value': range(50)}, index=dates)
        
        splitter = TSInSampleSplit(test_size=10, max_train_size=25)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        
        # Training size should respect max_train_size
        assert len(train_idx) <= 25
        # But should still include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_insample_max_train_size_smaller_than_test(self):
        """Test in-sample split where max_train_size < test_size."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({'value': range(30)}, index=dates)
        
        splitter = TSInSampleSplit(test_size=15, max_train_size=10)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        
        # Training should be at least as large as test (since test is included)
        assert len(train_idx) >= len(test_idx)
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_groups_parameter_warning(self):
        """Test that groups parameter issues warning and is ignored."""
        X = pd.Series(range(10))
        groups = np.array(['A'] * 5 + ['B'] * 5)
        
        splitter = TSInSampleSplit(test_size=3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splits = list(splitter.split(X, groups=groups))
            
            # Should issue warning about ignoring groups
            assert len(w) == 1
            assert "groups parameter is ignored" in str(w[0].message)
            assert "Time series splits treat data as a single temporal sequence" in str(w[0].message)
        
        # Should still work normally
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(test_idx) == 3
        assert np.all(np.isin(test_idx, train_idx))

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X = np.arange(20)
        
        splitter = TSInSampleSplit(test_size=5)
        splits = list(splitter.split(X))
        
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(test_idx) == 5
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_single_observation(self):
        """Test with single observation."""
        X = pd.Series([42])
        
        splitter = TSInSampleSplit(test_size=1)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 1
        assert len(test_idx) == 1
        assert train_idx[0] == test_idx[0]

    def test_empty_series(self):
        """Test with empty series."""
        X = pd.Series([], dtype=float)
        
        splitter = TSInSampleSplit(test_size=1)
        
        with pytest.raises(ValueError):
            list(splitter.split(X))

    def test_y_parameter_handling(self):
        """Test proper handling of y parameter."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        X = pd.DataFrame({'feature': range(20)}, index=dates)
        y = pd.Series(range(20, 40), index=dates)
        
        splitter = TSInSampleSplit(test_size=5)
        splits = list(splitter.split(X, y))
        
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        
        # Verify we can use indices to slice both X and y
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_default_test_size_calculation(self):
        """Test default test size calculation when not specified."""
        X = pd.Series(range(21))  # 21 observations
        
        splitter = TSInSampleSplit()  # Default n_splits=5, no test_size
        splits = list(splitter.split(X))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        
        # Default test size should be reasonable portion of data
        expected_test_size = 21 // (5 + 1)  # n_samples // (n_splits + 1)
        assert len(test_idx) == expected_test_size

    def test_invalid_test_dates(self):
        """Test with invalid test dates."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates)
        
        test_dates = ['2020-02-01']  # Not in the series
        splitter = TSInSampleSplit(test_indices=test_dates, test_size=1)
        
        with pytest.raises(ValueError, match="not found in data"):
            list(splitter.split(X))