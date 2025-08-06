"""Integration tests for the crossvals module.

This module tests the integration between different components and
real-world usage scenarios.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from tsforecast.crossvals import (
    OutOfSampleSplit,
    InSampleSplit,
    TSOutOfSampleSplit,
    TSInSampleSplit,
    PanelOutOfSampleSplit,
    PanelInSampleSplit,
    PanelOutOfSampleSplitPerEntity,
    PanelInSampleSplitPerEntity
)


class TestCrossValidationWorkflows:
    """Test realistic cross-validation workflows."""

    def test_time_series_forecasting_workflow(self, simple_timeseries_df):
        """Test a typical time series forecasting workflow."""
        X = simple_timeseries_df[['feature1', 'feature2']]
        y = simple_timeseries_df['target']
        
        # Test out-of-sample validation with rolling window
        splitter = TSOutOfSampleSplit(n_splits=5, test_size=3, gap=1, max_train_size=15)
        
        rmse_scores = []
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple linear model for testing
            # y = feature1 * coef + intercept
            coef = np.mean(y_train / X_train['feature1'])
            y_pred = X_test['feature1'] * coef
            
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            rmse_scores.append(rmse)
        
        assert len(rmse_scores) == 5
        assert all(score >= 0 for score in rmse_scores)

    def test_panel_data_forecasting_workflow(self, simple_panel):
        """Test a typical panel data forecasting workflow."""
        X = simple_panel[['feature1', 'feature2']]
        y = simple_panel['target']
        
        # Test out-of-sample validation
        splitter = PanelOutOfSampleSplit(n_splits=3, test_size=2, gap=1)
        
        entity_rmse = {}
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple model per entity
            for entity in X_test.index.get_level_values(0).unique():
                entity_mask_train = X_train.index.get_level_values(0) == entity
                entity_mask_test = X_test.index.get_level_values(0) == entity
                
                if entity_mask_train.any() and entity_mask_test.any():
                    X_entity_train = X_train[entity_mask_train]
                    y_entity_train = y_train[entity_mask_train]
                    X_entity_test = X_test[entity_mask_test]
                    y_entity_test = y_test[entity_mask_test]
                    
                    # Simple linear model
                    coef = np.mean(y_entity_train / X_entity_train['feature1'])
                    y_pred = X_entity_test['feature1'] * coef
                    
                    rmse = np.sqrt(np.mean((y_entity_test - y_pred) ** 2))
                    
                    if entity not in entity_rmse:
                        entity_rmse[entity] = []
                    entity_rmse[entity].append(rmse)
        
        # Should have scores for each entity
        assert len(entity_rmse) >= 2
        for entity_scores in entity_rmse.values():
            assert all(score >= 0 for score in entity_scores)

    def test_per_entity_validation_workflow(self, simple_panel):
        """Test per-entity validation workflow."""
        X = simple_panel[['feature1', 'feature2']]
        y = simple_panel['target']
        
        # Test per-entity out-of-sample validation
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=3, gap=1)
        
        entity_results = {}
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Get entity for this split
            entity = X_test.index.get_level_values(0).unique()[0]
            
            # Simple model
            coef = np.mean(y_train / X_train['feature1'])
            y_pred = X_test['feature1'] * coef
            
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            
            if entity not in entity_results:
                entity_results[entity] = []
            entity_results[entity].append(rmse)
        
        # Should have results for each entity
        expected_entities = simple_panel.index.get_level_values(0).unique()
        for entity in expected_entities:
            assert entity in entity_results
            # Each entity should have 2 splits (n_splits=2)
            assert len(entity_results[entity]) == 2

    def test_historical_backtesting_workflow(self, simple_timeseries_df):
        """Test historical backtesting with in-sample validation."""
        X = simple_timeseries_df[['feature1', 'feature2']]
        y = simple_timeseries_df['target']
        
        # Test in-sample validation
        test_dates = ['2020-01-15', '2020-01-25']
        splitter = TSInSampleSplit(test_indices=test_dates, test_size=3)
        
        results = []
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Model has access to future information (in-sample)
            coef = np.mean(y_train / X_train['feature1'])
            y_pred = X_test['feature1'] * coef
            
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            results.append(rmse)
        
        # Should have one result (in-sample typically produces one split)
        assert len(results) == 1


class TestDataTypeCompatibility:
    """Test compatibility with different data types and structures."""

    def test_datetime_index_compatibility(self):
        """Test compatibility with different datetime index types."""
        # Test different datetime frequencies
        for freq in ['D', 'H', 'M', 'Y']:
            if freq == 'Y':
                periods = 10
            elif freq == 'M':
                periods = 24
            else:
                periods = 50
                
            dates = pd.date_range('2020-01-01', periods=periods, freq=freq)
            X = pd.DataFrame({'feature': range(periods)}, index=dates)
            
            splitter = TSOutOfSampleSplit(n_splits=2, test_size=3)
            splits = list(splitter.split(X))
            
            assert len(splits) == 2
            for train_idx, test_idx in splits:
                assert len(test_idx) == 3

    def test_string_index_compatibility(self):
        """Test compatibility with string indices."""
        X = pd.DataFrame({
            'feature': range(20)
        }, index=[f'period_{i:02d}' for i in range(20)])
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=2)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(test_idx) == 2

    def test_numeric_entity_names(self):
        """Test compatibility with numeric entity names in panel data."""
        entities = [1, 2, 3, 4]
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(60)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=2)
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        for train_idx, test_idx in splits:
            # Should have test data for all entities
            test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
            assert len(test_entities) == 4

    def test_mixed_data_types_in_features(self):
        """Test with mixed data types in features."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        X = pd.DataFrame({
            'numeric_feature': range(30),
            'float_feature': np.random.randn(30),
            'categorical_feature': ['A', 'B', 'C'] * 10,
            'boolean_feature': [True, False] * 15
        }, index=dates)
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=5)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Verify all data types are preserved
            assert X_train.dtypes.equals(X.dtypes)
            assert X_test.dtypes.equals(X.dtypes)


class TestEdgeCaseHandling:
    """Test handling of edge cases and boundary conditions."""

    def test_very_small_datasets(self):
        """Test with very small datasets."""
        # Test with 3 observations
        X = pd.Series([1, 2, 3])
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=1)
        splits = list(splitter.split(X))
        
        # With n_splits=2 and 3 observations, only one split may be valid
        assert len(splits) >= 1
        train_idx, test_idx = splits[0]
        assert len(test_idx) == 1
        assert len(train_idx) <= 2

    def test_gap_edge_cases(self):
        """Test edge cases with gap parameter."""
        X = pd.Series(range(10))
        
        # Gap equal to available training data
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2, gap=7)
        splits = list(splitter.split(X))
        
        # Should result in minimal or empty training sets
        for train_idx, test_idx in splits:
            assert len(train_idx) <= 1

    def test_max_train_size_edge_cases(self):
        """Test edge cases with max_train_size parameter."""
        X = pd.Series(range(20))
        
        # max_train_size smaller than test_size
        splitter = InSampleSplit(test_size=10, max_train_size=5)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        # Training should still include test period but respect constraints
        assert np.all(np.isin(test_idx, train_idx))
        assert len(train_idx) >= len(test_idx)

    def test_single_entity_panel_edge_case(self):
        """Test edge case with single entity in panel data."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([['single_entity'], dates], names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(20)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=3, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        # Should behave like time series for single entity
        for train_idx, test_idx in splits:
            assert len(test_idx) == 3
            if len(train_idx) > 0:
                assert max(train_idx) < min(test_idx)


class TestPerformanceAndScalability:
    """Test performance with larger datasets."""

    @pytest.mark.slow
    def test_large_timeseries_performance(self, large_timeseries):
        """Test performance with large time series."""
        X = large_timeseries
        
        splitter = TSOutOfSampleSplit(n_splits=10, test_size=50, gap=5)
        
        import time
        start_time = time.time()
        splits = list(splitter.split(X))
        end_time = time.time()
        
        # Should complete reasonably quickly (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(splits) == 10
        
        for train_idx, test_idx in splits:
            assert len(test_idx) == 50

    @pytest.mark.slow
    def test_large_panel_performance(self, large_panel):
        """Test performance with large panel data."""
        X = large_panel
        
        splitter = PanelOutOfSampleSplit(n_splits=5, test_size=10, gap=2)
        
        import time
        start_time = time.time()
        splits = list(splitter.split(X))
        end_time = time.time()
        
        # Should complete reasonably quickly (less than 2 seconds)
        assert end_time - start_time < 2.0
        assert len(splits) == 5

    @pytest.mark.slow
    def test_per_entity_scaling(self, large_panel):
        """Test per-entity splitting with many entities."""
        X = large_panel
        
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=5)
        
        import time
        start_time = time.time()
        splits = list(splitter.split(X))
        end_time = time.time()
        
        # Should scale reasonably with number of entities
        n_entities = X.index.get_level_values(0).nunique()
        expected_splits = 3 * n_entities  # n_splits × n_entities
        
        assert len(splits) == expected_splits
        assert end_time - start_time < 3.0


class TestErrorHandlingAndValidation:
    """Test comprehensive error handling and input validation."""

    def test_comprehensive_parameter_validation(self):
        """Test validation of all parameter combinations."""
        X = pd.Series(range(20))
        
        # Test negative parameters
        with pytest.raises(ValueError):
            splitter = TSOutOfSampleSplit(n_splits=-1)
            list(splitter.split(X))
        
        # Test impossible parameter combinations
        with pytest.raises(ValueError):
            splitter = TSOutOfSampleSplit(n_splits=10, test_size=5)
            list(splitter.split(X))

    def test_index_validation_comprehensive(self):
        """Test comprehensive index validation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        X = pd.Series(range(10), index=dates)
        
        # Test various invalid test_indices
        invalid_indices = [
            ['2019-12-31'],  # Before data start
            ['2020-01-15'],  # After data end
            [pd.Timestamp('2019-12-31')],  # Timestamp before data
            [-1, -2],        # Negative positions
            [15, 20],        # Positions beyond data
        ]
        
        for test_indices in invalid_indices:
            splitter = TSOutOfSampleSplit(test_indices=test_indices, test_size=1)
            with pytest.raises(ValueError):
                list(splitter.split(X))

    def test_data_structure_validation_comprehensive(self):
        """Test comprehensive data structure validation."""
        # Test various invalid panel structures
        invalid_panels = [
            pd.DataFrame({'value': range(10)}),  # No MultiIndex
            pd.Series(range(10)),               # Series without MultiIndex for panel
        ]
        
        for X in invalid_panels:
            splitter = PanelOutOfSampleSplit(n_splits=2, test_size=1)
            with pytest.raises(ValueError):
                list(splitter.split(X))

    def test_warning_generation(self):
        """Test that appropriate warnings are generated."""
        # Test groups parameter warning for time series
        X = pd.Series(range(10))
        groups = np.array(['A'] * 5 + ['B'] * 5)
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(splitter.split(X, groups=groups))
            
            assert len(w) == 1
            assert "groups parameter is ignored" in str(w[0].message)

    def test_data_sorting_warnings(self, unsorted_timeseries, unsorted_panel):
        """Test warnings for unsorted data."""
        # Test unsorted time series
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(splitter.split(unsorted_timeseries))
            
            assert len(w) == 1
            assert "not sorted by date" in str(w[0].message)
        
        # Test unsorted panel
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(splitter.split(unsorted_panel))
            
            assert len(w) == 1
            assert "not sorted by group then by date" in str(w[0].message)


class TestConsistencyAndReproducibility:
    """Test consistency and reproducibility of splits."""

    def test_split_consistency(self, simple_timeseries_df):
        """Test that splits are consistent across multiple calls."""
        X = simple_timeseries_df
        
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=5, gap=2)
        
        # Generate splits multiple times
        splits1 = list(splitter.split(X))
        splits2 = list(splitter.split(X))
        
        # Should be identical
        assert len(splits1) == len(splits2)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_parameter_independence(self, simple_timeseries_df):
        """Test that different splitter instances with same parameters produce same results."""
        X = simple_timeseries_df
        
        splitter1 = TSOutOfSampleSplit(n_splits=3, test_size=4, gap=1)
        splitter2 = TSOutOfSampleSplit(n_splits=3, test_size=4, gap=1)
        
        splits1 = list(splitter1.split(X))
        splits2 = list(splitter2.split(X))
        
        assert len(splits1) == len(splits2)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_data_modification_independence(self, simple_timeseries_df):
        """Test that modifying data after split generation doesn't affect splits."""
        X = simple_timeseries_df.copy()
        
        splitter = TSOutOfSampleSplit(n_splits=2, test_size=3)
        splits_before = list(splitter.split(X))
        
        # Modify the data
        X.iloc[0, 0] = 999999
        
        splits_after = list(splitter.split(X))
        
        # Splits should be different due to data change, but structure should be same
        assert len(splits_before) == len(splits_after)
        for (train1, test1), (train2, test2) in zip(splits_before, splits_after):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)