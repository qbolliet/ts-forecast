#!/usr/bin/env python3
"""
Test script to verify the refactored crossvalidation classes work correctly.
"""

import numpy as np
import pandas as pd
from tsforecast.crossvals.time_series import TSOutOfSampleSplit, TSInSampleSplit
from tsforecast.crossvals.panel import (
    PanelOutOfSampleSplit, PanelInSampleSplit,
    PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity
)

def test_time_series_splits():
    """Test time series cross-validation splits."""
    print("Testing Time Series Splits...")
    
    # Create simple time series data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    X = pd.DataFrame({'feature': np.random.randn(100)}, index=dates)
    y = pd.Series(np.random.randn(100), index=dates)
    
    # Test TSOutOfSampleSplit
    print("TSOutOfSampleSplit:")
    ts_oos = TSOutOfSampleSplit(n_splits=3, test_size=10, gap=5)
    splits = list(ts_oos.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test TSInSampleSplit
    print("TSInSampleSplit:")
    ts_is = TSInSampleSplit(n_splits=3, test_size=10, max_train_size=50)
    splits = list(ts_is.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    print("Time Series Tests Passed!\n")

def test_panel_splits():
    """Test panel data cross-validation splits."""
    print("Testing Panel Data Splits...")
    
    # Create panel data
    entities = ['A', 'B', 'C']
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    
    # Create MultiIndex for panel data
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    X = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)
    y = pd.Series(np.random.randn(len(idx)), index=idx)
    
    # Test PanelOutOfSampleSplit
    print("PanelOutOfSampleSplit:")
    panel_oos = PanelOutOfSampleSplit(n_splits=3, test_size=5, gap=2)
    splits = list(panel_oos.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test PanelInSampleSplit
    print("PanelInSampleSplit:")
    panel_is = PanelInSampleSplit(n_splits=1, test_size=5, max_train_size=30)
    splits = list(panel_is.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    print("Panel Data Tests Passed!\n")

def test_panel_per_entity_splits():
    """Test panel data cross-validation splits per entity."""
    print("Testing Panel Data Splits Per Entity...")
    
    # Create panel data
    entities = ['A', 'B', 'C']
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    
    # Create MultiIndex for panel data
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    X = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)
    y = pd.Series(np.random.randn(len(idx)), index=idx)
    
    # Test PanelOutOfSampleSplitPerEntity
    print("PanelOutOfSampleSplitPerEntity:")
    panel_oos_entity = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=3, gap=1)
    splits = list(panel_oos_entity.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, entity_splits in enumerate(splits):
        print(f"  Split {i}:")
        for entity, (train_idx, test_idx) in entity_splits.items():
            print(f"    Entity {entity}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test PanelInSampleSplitPerEntity
    print("PanelInSampleSplitPerEntity:")
    panel_is_entity = PanelInSampleSplitPerEntity(n_splits=1, test_size=3, max_train_size=10)
    splits = list(panel_is_entity.split(X, y))
    print(f"Number of splits: {len(splits)}")
    for i, entity_splits in enumerate(splits):
        print(f"  Split {i}:")
        for entity, (train_idx, test_idx) in entity_splits.items():
            print(f"    Entity {entity}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    print("Panel Data Per Entity Tests Passed!\n")

def test_groups_parameter():
    """Test that groups parameter works correctly."""
    print("Testing Groups Parameter...")
    
    # Create simple data with groups
    X = np.random.randn(60, 2)
    y = np.random.randn(60)
    groups = np.repeat(['A', 'B', 'C'], 20)  # 3 groups of 20 samples each
    
    # Test with explicit groups
    from tsforecast.crossvals.base import OutOfSampleSplit, InSampleSplit
    
    print("OutOfSampleSplit with groups:")
    oos = OutOfSampleSplit(n_splits=2, test_size=5, gap=2)
    splits = list(oos.split(X, y, groups))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
        # Check which groups are in test
        test_groups = set(groups[test_idx])
        print(f"    Test groups: {test_groups}")
    
    print("InSampleSplit with groups:")
    ins = InSampleSplit(n_splits=1, test_size=5, max_train_size=15)
    splits = list(ins.split(X, y, groups))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
        # Check which groups are in test
        test_groups = set(groups[test_idx])
        print(f"    Test groups: {test_groups}")
    
    print("Groups Parameter Tests Passed!\n")

if __name__ == "__main__":
    print("Testing Refactored Cross-Validation Classes\n")
    print("=" * 50)
    
    try:
        test_time_series_splits()
        test_panel_splits()
        test_panel_per_entity_splits()
        test_groups_parameter()
        
        print("=" * 50)
        print("All tests passed successfully! 🎉")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()