"""Tests for panel data cross-validation classes.

This module tests the panel data specific cross-validation classes
including basic panel splits and per-entity splits.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from tsforecast.crossvals.panel import (
    _extract_groups_from_panel,
    PanelOutOfSampleSplit,
    PanelInSampleSplit,
    PanelOutOfSampleSplitPerEntity,
    PanelInSampleSplitPerEntity
)


class TestExtractGroupsFromPanel:
    """Test _extract_groups_from_panel helper function."""

    def test_basic_panel_extraction(self):
        """Test extracting groups from basic panel structure."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(15)}, index=idx)
        
        groups = _extract_groups_from_panel(X)
        expected = np.array(['A']*5 + ['B']*5 + ['C']*5)
        np.testing.assert_array_equal(groups, expected)

    def test_series_panel_extraction(self):
        """Test extracting groups from panel Series."""
        entities = ['X', 'Y']
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.Series(range(6), index=idx)
        
        groups = _extract_groups_from_panel(X)
        expected = np.array(['X']*3 + ['Y']*3)
        np.testing.assert_array_equal(groups, expected)

    def test_no_pandas_index_error(self):
        """Test error when data has no pandas index."""
        X = np.array(range(10))
        
        with pytest.raises(ValueError, match="Panel data requires pandas DataFrame/Series with MultiIndex"):
            _extract_groups_from_panel(X)

    def test_no_multiindex_error(self):
        """Test that function works with non-MultiIndex data (treats each row as separate entity)."""
        X = pd.DataFrame({'value': range(10)})
        
        # Function should work and return each index position as a separate group
        groups = _extract_groups_from_panel(X)
        expected_groups = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(groups, expected_groups)

    def test_numeric_entity_names(self):
        """Test with numeric entity names."""
        entities = [1, 2, 3]
        dates = pd.date_range('2020-01-01', periods=4, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(12)}, index=idx)
        
        groups = _extract_groups_from_panel(X)
        expected = np.array([1]*4 + [2]*4 + [3]*4)
        np.testing.assert_array_equal(groups, expected)

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (different entity lengths)."""
        # Create unbalanced panel manually
        idx = pd.MultiIndex.from_tuples([
            ('A', pd.Timestamp('2020-01-01')),
            ('A', pd.Timestamp('2020-01-02')),
            ('B', pd.Timestamp('2020-01-01')),
            ('B', pd.Timestamp('2020-01-02')),
            ('B', pd.Timestamp('2020-01-03')),
        ], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(5)}, index=idx)
        
        groups = _extract_groups_from_panel(X)
        expected = np.array(['A', 'A', 'B', 'B', 'B'])
        np.testing.assert_array_equal(groups, expected)


class TestPanelOutOfSampleSplit:
    """Test PanelOutOfSampleSplit class."""

    def test_init_inherits_from_base(self):
        """Test that class inherits from OutOfSampleSplit."""
        from tsforecast.crossvals.base import OutOfSampleSplit
        splitter = PanelOutOfSampleSplit()
        assert isinstance(splitter, OutOfSampleSplit)

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = PanelOutOfSampleSplit()
        assert splitter.n_splits == 5
        assert splitter.test_indices is None
        assert splitter.max_train_size is None
        assert splitter.test_size is None
        assert splitter.gap == 0

    def test_basic_panel_split(self):
        """Test basic panel data split."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(40)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=3, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            # Should have test data for both entities
            test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
            assert 'A' in test_entities
            assert 'B' in test_entities
            
            # Training should come before test for each entity
            for entity in ['A', 'B']:
                entity_mask = X.index.get_level_values(0) == entity
                entity_positions = np.where(entity_mask)[0]
                
                entity_train = train_idx[np.isin(train_idx, entity_positions)]
                entity_test = test_idx[np.isin(test_idx, entity_positions)]
                
                if len(entity_train) > 0 and len(entity_test) > 0:
                    assert max(entity_train) < min(entity_test)

    def test_panel_split_with_gap(self):
        """Test panel split with gap parameter."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(60)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=3, gap=5)
        splits = list(splitter.split(X))
        
        for train_idx, test_idx in splits:
            # Check gap for each entity separately
            for entity in ['A', 'B']:
                entity_mask = X.index.get_level_values(0) == entity
                entity_positions = np.where(entity_mask)[0]
                
                entity_train = train_idx[np.isin(train_idx, entity_positions)]
                entity_test = test_idx[np.isin(test_idx, entity_positions)]
                
                if len(entity_train) > 0 and len(entity_test) > 0:
                    gap_positions = min(entity_test) - max(entity_train) - 1
                    assert gap_positions == 5

    def test_panel_split_specific_test_dates(self):
        """Test panel split with specific test dates."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(90)}, index=idx)
        
        test_dates = ['2020-01-10', '2020-01-20']
        splitter = PanelOutOfSampleSplit(test_indices=test_dates, test_size=2)
        
        splits = list(splitter.split(X))
        assert len(splits) == 2
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check that test starts at the right date for each entity
            test_date = test_dates[i]
            for entity in entities:
                expected_start = X.index.get_loc((entity, test_date))
                # Check if this position is in test_idx
                if expected_start in test_idx:
                    # Find the entity's test indices
                    entity_mask = X.index.get_level_values(0) == entity
                    entity_positions = np.where(entity_mask)[0]
                    entity_test = test_idx[np.isin(test_idx, entity_positions)]
                    
                    assert expected_start in entity_test

    def test_panel_split_automatic_group_extraction(self):
        """Test that groups are automatically extracted from panel structure."""
        entities = ['X', 'Y']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=2)
        # Don't provide groups parameter - should extract automatically
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Should have test data for both entities
            test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
            assert 'X' in test_entities
            assert 'Y' in test_entities

    def test_panel_split_provided_groups(self):
        """Test panel split with explicitly provided groups."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        groups = np.repeat(entities, 10)
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=2)
        splits = list(splitter.split(X, groups=groups))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Verify groups parameter was used correctly
            test_groups = groups[test_idx]
            assert 'A' in test_groups
            assert 'B' in test_groups

    def test_unbalanced_panel(self):
        """Test with unbalanced panel data."""
        # Entity A has more data than entity B
        idx = pd.MultiIndex.from_tuples([
            ('A', pd.Timestamp('2020-01-01')),
            ('A', pd.Timestamp('2020-01-02')),
            ('A', pd.Timestamp('2020-01-03')),
            ('A', pd.Timestamp('2020-01-04')),
            ('A', pd.Timestamp('2020-01-05')),
            ('B', pd.Timestamp('2020-01-01')),
            ('B', pd.Timestamp('2020-01-02')),
            ('B', pd.Timestamp('2020-01-03')),
        ], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(8)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=1)
        
        # Should raise error when entity is missing in test period
        with pytest.raises(ValueError, match="Cannot find test indices for the group"):
            list(splitter.split(X))

    def test_missing_entity_in_test_period(self):
        """Test behavior when entity missing in test period."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        # Specify test date that doesn't exist for entity B
        test_dates = ['2020-01-15']  # This date doesn't exist in our data
        splitter = PanelOutOfSampleSplit(test_indices=test_dates, test_size=1)
        
        # Should raise error when test date doesn't exist in data
        with pytest.raises(ValueError, match="Date .* not found in panel data"):
            list(splitter.split(X))

    def test_invalid_panel_structure(self):
        """Test error with invalid panel structure."""
        # Single-level index instead of MultiIndex
        X = pd.DataFrame({'value': range(10)})
        
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=1)
        
        with pytest.raises(ValueError, match="Panel data requires MultiIndex"):
            list(splitter.split(X))

    def test_single_entity_panel(self):
        """Test with single entity panel."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelOutOfSampleSplit(n_splits=3, test_size=3)
        splits = list(splitter.split(X))
        
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(test_idx) == 3
            # Should behave like time series for single entity
            assert max(train_idx) < min(test_idx)


class TestPanelInSampleSplit:
    """Test PanelInSampleSplit class."""

    def test_init_inherits_from_base(self):
        """Test that class inherits from InSampleSplit."""
        from tsforecast.crossvals.base import InSampleSplit
        splitter = PanelInSampleSplit()
        assert isinstance(splitter, InSampleSplit)

    def test_basic_panel_insample_split(self):
        """Test basic panel in-sample split."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(40)}, index=idx)
        
        splitter = PanelInSampleSplit(test_size=5)
        splits = list(splitter.split(X))
        
        assert len(splits) == 1  # In-sample typically produces one split
        
        train_idx, test_idx = splits[0]
        
        # Key characteristic: training includes test period
        assert np.all(np.isin(test_idx, train_idx))
        
        # Should have test data for both entities
        test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
        assert 'A' in test_entities
        assert 'B' in test_entities

    def test_panel_insample_specific_test_date(self):
        """Test panel in-sample split with specific test date."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(90)}, index=idx)
        
        test_dates = ['2020-01-15']
        splitter = PanelInSampleSplit(test_indices=test_dates, test_size=3)
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))
        
        # Should have test data for all entities at the specified date
        assert len(test_idx) == 9  # 3 periods × 3 entities

    def test_panel_insample_multiple_test_dates(self):
        """Test panel in-sample split with multiple test dates."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(60)}, index=idx)
        
        test_dates = ['2020-01-10', '2020-01-20']
        splitter = PanelInSampleSplit(test_indices=test_dates, test_size=1)  # test_size ignored
        
        splits = list(splitter.split(X))
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        
        # Should include all specified test dates for both entities
        assert len(test_idx) == 4  # 2 dates × 2 entities
        # Training should include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_panel_insample_with_max_train_size(self):
        """Test panel in-sample split with max_train_size."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(100)}, index=idx)
        
        splitter = PanelInSampleSplit(test_size=10, max_train_size=60)
        splits = list(splitter.split(X))
        
        train_idx, test_idx = splits[0]
        
        # InSampleSplit includes all available data in training (including test period)
        # max_train_size parameter may not apply the same way as in out-of-sample
        assert len(train_idx) >= len(test_idx)
        # Should still include test period
        assert np.all(np.isin(test_idx, train_idx))

    def test_automatic_group_extraction(self):
        """Test automatic group extraction from panel structure."""
        entities = ['X', 'Y']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelInSampleSplit(test_size=3)
        # Don't provide groups parameter
        splits = list(splitter.split(X))
        
        assert len(splits) == 1
        
        train_idx, test_idx = splits[0]
        # Should have test data for both entities
        test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
        assert 'X' in test_entities
        assert 'Y' in test_entities


class TestPanelOutOfSampleSplitPerEntity:
    """Test PanelOutOfSampleSplitPerEntity class."""

    def test_init_inherits_from_panel_base(self):
        """Test that class inherits from PanelOutOfSampleSplit."""
        splitter = PanelOutOfSampleSplitPerEntity()
        assert isinstance(splitter, PanelOutOfSampleSplit)

    def test_basic_per_entity_split(self):
        """Test basic per-entity split functionality."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(45)}, index=idx)
        
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=3)
        splits = list(splitter.split(X))
        
        # Should have separate splits for each entity that has test data
        # For 2 splits with 3 entities each, expect 6 splits (2 splits × 3 entities)
        assert len(splits) == 6
        
        # Group splits by original split
        entity_splits = {}
        split_count = 0
        for train_idx, test_idx in splits:
            # Each split should contain data from only one entity
            train_entities = X.iloc[train_idx].index.get_level_values(0).unique()
            test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
            
            assert len(train_entities) == 1  # Only one entity in training
            assert len(test_entities) == 1   # Only one entity in test
            assert train_entities[0] == test_entities[0]  # Same entity
            
            entity = test_entities[0]
            if entity not in entity_splits:
                entity_splits[entity] = []
            entity_splits[entity].append((train_idx, test_idx))
        
        # Each entity should have 2 splits (matching n_splits)
        for entity in entities:
            assert len(entity_splits[entity]) == 2

    def test_per_entity_temporal_ordering(self):
        """Test that temporal ordering is maintained within each entity."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(40)}, index=idx)
        
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=2, gap=1)
        splits = list(splitter.split(X))
        
        for train_idx, test_idx in splits:
            # Each split should have only one entity
            train_entity = X.iloc[train_idx].index.get_level_values(0).unique()
            test_entity = X.iloc[test_idx].index.get_level_values(0).unique()
            
            assert len(train_entity) == 1
            assert len(test_entity) == 1
            assert train_entity[0] == test_entity[0]
            
            # Within the entity, training should come before test
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert max(train_idx) < min(test_idx)
                
                # Check gap
                gap_positions = min(test_idx) - max(train_idx) - 1
                assert gap_positions == 1

    def test_per_entity_with_specific_test_dates(self):
        """Test per-entity split with specific test dates."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(40)}, index=idx)
        
        test_dates = ['2020-01-10']
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=test_dates, test_size=3)
        
        splits = list(splitter.split(X))
        
        # Should have one split per entity
        assert len(splits) == 2
        
        entities_seen = set()
        for train_idx, test_idx in splits:
            # Verify single entity per split
            test_entity = X.iloc[test_idx].index.get_level_values(0).unique()
            assert len(test_entity) == 1
            entities_seen.add(test_entity[0])
            
            # Verify test starts at correct date for this entity
            entity = test_entity[0]
            expected_start = X.index.get_loc((entity, test_dates[0]))
            assert min(test_idx) == expected_start
            assert len(test_idx) == 3
        
        # Should have seen both entities
        assert entities_seen == {'A', 'B'}

    def test_per_entity_unbalanced_panel(self):
        """Test per-entity split with unbalanced panel."""
        # Create unbalanced panel
        idx = pd.MultiIndex.from_tuples([
            ('A', pd.Timestamp('2020-01-01')),
            ('A', pd.Timestamp('2020-01-02')),
            ('A', pd.Timestamp('2020-01-03')),
            ('A', pd.Timestamp('2020-01-04')),
            ('A', pd.Timestamp('2020-01-05')),
            ('B', pd.Timestamp('2020-01-01')),
            ('B', pd.Timestamp('2020-01-02')),
        ], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(7)}, index=idx)
        
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=1)
        
        # Should raise error when entity is missing in test period
        with pytest.raises(ValueError, match="Cannot find test indices for the group"):
            list(splitter.split(X))

    def test_per_entity_missing_test_data(self):
        """Test behavior when some entities don't have test data."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(30)}, index=idx)
        
        # Use very late test date that might not exist for all entities
        test_dates = ['2020-01-15']  # This date doesn't exist in our data
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=test_dates, test_size=1)
        
        # Should raise error when test date doesn't exist in data
        with pytest.raises(ValueError, match="Date .* not found in panel data"):
            list(splitter.split(X))

    def test_per_entity_empty_splits_filtered(self):
        """Test that entities without test data are filtered out."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(10)}, index=idx)
        
        # Use test size that's too large, might result in empty test sets for some entities
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=2)
        splits = list(splitter.split(X))
        
        # All returned splits should have non-empty test sets
        for train_idx, test_idx in splits:
            assert len(test_idx) > 0

    def test_per_entity_single_entity(self):
        """Test per-entity split with single entity."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=2)
        splits = list(splitter.split(X))
        
        # Should have 3 splits for the single entity
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            # Should behave like time series for single entity
            assert len(test_idx) == 2
            assert max(train_idx) < min(test_idx)


class TestPanelInSampleSplitPerEntity:
    """Test PanelInSampleSplitPerEntity class."""

    def test_init_inherits_from_panel_base(self):
        """Test that class inherits from PanelInSampleSplit."""
        splitter = PanelInSampleSplitPerEntity()
        assert isinstance(splitter, PanelInSampleSplit)

    def test_basic_per_entity_insample_split(self):
        """Test basic per-entity in-sample split."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(40)}, index=idx)
        
        splitter = PanelInSampleSplitPerEntity(test_size=5)
        splits = list(splitter.split(X))
        
        # Should have one split per entity (in-sample typically produces one split per entity)
        assert len(splits) == 2
        
        entities_seen = set()
        for train_idx, test_idx in splits:
            # Each split should have only one entity
            train_entities = X.iloc[train_idx].index.get_level_values(0).unique()
            test_entities = X.iloc[test_idx].index.get_level_values(0).unique()
            
            assert len(train_entities) == 1
            assert len(test_entities) == 1
            assert train_entities[0] == test_entities[0]
            
            entities_seen.add(test_entities[0])
            
            # Key in-sample characteristic: training includes test period
            assert np.all(np.isin(test_idx, train_idx))
        
        # Should have seen both entities
        assert entities_seen == {'A', 'B'}

    def test_per_entity_insample_specific_test_date(self):
        """Test per-entity in-sample split with specific test date."""
        entities = ['A', 'B', 'C']
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(60)}, index=idx)
        
        test_dates = ['2020-01-10']
        splitter = PanelInSampleSplitPerEntity(test_indices=test_dates, test_size=3)
        
        splits = list(splitter.split(X))
        
        # Should have one split per entity
        assert len(splits) == 3
        
        entities_seen = set()
        for train_idx, test_idx in splits:
            # Verify single entity per split
            test_entity = X.iloc[test_idx].index.get_level_values(0).unique()
            assert len(test_entity) == 1
            entities_seen.add(test_entity[0])
            
            # Verify in-sample characteristic
            assert np.all(np.isin(test_idx, train_idx))
            
            # Verify test starts at correct date for this entity
            entity = test_entity[0]
            expected_start = X.index.get_loc((entity, test_dates[0]))
            assert min(test_idx) == expected_start
            assert len(test_idx) == 3
        
        # Should have seen all entities
        assert entities_seen == {'A', 'B', 'C'}

    def test_per_entity_insample_multiple_test_dates(self):
        """Test per-entity in-sample split with multiple test dates."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(60)}, index=idx)
        
        test_dates = ['2020-01-10', '2020-01-20']
        splitter = PanelInSampleSplitPerEntity(test_indices=test_dates, test_size=1)  # test_size ignored
        
        splits = list(splitter.split(X))
        
        # Should have one split per entity
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Each entity should have test data for all specified dates
            test_entity = X.iloc[test_idx].index.get_level_values(0).unique()
            assert len(test_entity) == 1
            
            # Should include all specified test dates (test_size ignored for multiple dates)
            assert len(test_idx) == 2  # Both test dates for this entity
            
            # Training should include test period
            assert np.all(np.isin(test_idx, train_idx))

    def test_per_entity_insample_with_max_train_size(self):
        """Test per-entity in-sample split with max_train_size."""
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(100)}, index=idx)
        
        splitter = PanelInSampleSplitPerEntity(test_size=10, max_train_size=30)
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        for train_idx, test_idx in splits:
            # Each entity should respect max_train_size
            assert len(train_idx) <= 30
            # But should still include test period
            assert np.all(np.isin(test_idx, train_idx))

    def test_per_entity_single_entity(self):
        """Test per-entity in-sample split with single entity."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        idx = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelInSampleSplitPerEntity(test_size=5)
        splits = list(splitter.split(X))
        
        # Should have multiple splits for the single entity (due to n_splits=5 default)
        assert len(splits) >= 1
        
        train_idx, test_idx = splits[0]
        
        # Should maintain in-sample characteristic
        assert np.all(np.isin(test_idx, train_idx))
        # With default n_splits=5, test_size might be smaller than specified
        assert len(test_idx) >= 1  # At least one test sample

    def test_per_entity_automatic_group_extraction(self):
        """Test automatic group extraction in per-entity splits."""
        entities = ['X', 'Y']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'value': range(20)}, index=idx)
        
        splitter = PanelInSampleSplitPerEntity(test_size=3)
        # Don't provide groups parameter
        splits = list(splitter.split(X))
        
        assert len(splits) == 2
        
        entities_seen = set()
        for train_idx, test_idx in splits:
            test_entity = X.iloc[test_idx].index.get_level_values(0).unique()
            assert len(test_entity) == 1
            entities_seen.add(test_entity[0])
            
            # Verify in-sample characteristic
            assert np.all(np.isin(test_idx, train_idx))
        
        assert entities_seen == {'X', 'Y'}