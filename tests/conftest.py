"""Pytest configuration and fixtures for tsforecast tests.

This module provides common fixtures and configuration for all tests.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
# Tests
import pytest


@pytest.fixture
def simple_timeseries():
    """Simple time series data for testing."""
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    return pd.Series(range(30), index=dates, name='value')


@pytest.fixture
def simple_timeseries_df():
    """Simple time series DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'feature1': range(30),
        'feature2': np.sin(np.arange(30) * 0.1),
        'target': range(30, 60)
    }, index=dates)


@pytest.fixture
def simple_panel():
    """Simple panel data for testing."""
    entities = ['A', 'B', 'C']
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    return pd.DataFrame({
        'feature1': range(60),
        'feature2': np.random.randn(60),
        'target': range(60, 120)
    }, index=idx)


@pytest.fixture
def unbalanced_panel():
    """Unbalanced panel data for testing."""
    data = []
    indices = []
    
    # Entity A: 10 periods
    entity_a_dates = pd.date_range('2020-01-01', periods=10, freq='D')
    for i, date in enumerate(entity_a_dates):
        data.append([i, np.random.randn(), i + 100])
        indices.append(('A', date))
    
    # Entity B: 15 periods
    entity_b_dates = pd.date_range('2020-01-01', periods=15, freq='D')
    for i, date in enumerate(entity_b_dates):
        data.append([i + 10, np.random.randn(), i + 200])
        indices.append(('B', date))
    
    # Entity C: 8 periods
    entity_c_dates = pd.date_range('2020-01-01', periods=8, freq='D')
    for i, date in enumerate(entity_c_dates):
        data.append([i + 25, np.random.randn(), i + 300])
        indices.append(('C', date))
    
    idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
    return pd.DataFrame(data, columns=['feature1', 'feature2', 'target'], index=idx)


@pytest.fixture
def large_timeseries():
    """Large time series for performance testing."""
    dates = pd.date_range('2000-01-01', periods=1000, freq='D')
    return pd.DataFrame({
        'feature1': np.cumsum(np.random.randn(1000)),
        'feature2': np.sin(np.arange(1000) * 0.01),
        'feature3': np.random.exponential(1, 1000),
        'target': np.cumsum(np.random.randn(1000)) + np.random.randn(1000) * 0.1
    }, index=dates)


@pytest.fixture
def large_panel():
    """Large panel data for performance testing."""
    entities = [f'entity_{i:02d}' for i in range(10)]
    dates = pd.date_range('2010-01-01', periods=200, freq='D')
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    
    n_obs = len(entities) * len(dates)
    return pd.DataFrame({
        'feature1': np.random.randn(n_obs),
        'feature2': np.cumsum(np.random.randn(n_obs).reshape(len(entities), -1), axis=1).flatten(),
        'feature3': np.random.exponential(1, n_obs),
        'target': np.random.randn(n_obs) * 0.5 + np.arange(n_obs) * 0.01
    }, index=idx)


@pytest.fixture
def numpy_array_data():
    """Simple numpy array data for testing."""
    return np.random.randn(50, 3)


@pytest.fixture
def mismatched_xy():
    """X and y with mismatched indices for testing error handling."""
    dates_x = pd.date_range('2020-01-01', periods=20, freq='D')
    dates_y = pd.date_range('2020-01-02', periods=20, freq='D')  # Offset by 1 day
    
    X = pd.DataFrame({'feature': range(20)}, index=dates_x)
    y = pd.Series(range(20, 40), index=dates_y)
    
    return X, y


@pytest.fixture
def unsorted_timeseries():
    """Unsorted time series data for testing sorting functionality."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    # Shuffle the dates
    shuffled_indices = [2, 0, 7, 4, 1, 9, 3, 6, 8, 5]
    shuffled_dates = [dates[i] for i in shuffled_indices]
    values = [i * 10 for i in shuffled_indices]  # Values corresponding to original order
    
    return pd.Series(values, index=shuffled_dates, name='value')


@pytest.fixture
def unsorted_panel():
    """Unsorted panel data for testing sorting functionality."""
    # Create data with mixed entity and date order
    data_tuples = [
        ('B', pd.Timestamp('2020-01-02'), 10),
        ('A', pd.Timestamp('2020-01-01'), 0),
        ('B', pd.Timestamp('2020-01-01'), 5),
        ('A', pd.Timestamp('2020-01-03'), 15),
        ('A', pd.Timestamp('2020-01-02'), 12),
        ('B', pd.Timestamp('2020-01-03'), 20),
    ]
    
    indices = [(entity, date) for entity, date, _ in data_tuples]
    values = [value for _, _, value in data_tuples]
    
    idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
    return pd.DataFrame({'value': values}, index=idx)


@pytest.fixture(params=[
    'simple_timeseries',
    'simple_timeseries_df',
    'numpy_array_data'
])
def various_input_types(request):
    """Parametrized fixture providing various input data types."""
    if request.param == 'simple_timeseries':
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        return pd.Series(range(20), index=dates)
    elif request.param == 'simple_timeseries_df':
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        return pd.DataFrame({'feature': range(20)}, index=dates)
    elif request.param == 'numpy_array_data':
        return np.arange(20).reshape(-1, 1)


@pytest.fixture(params=[
    ('simple_panel', None),
    ('unbalanced_panel', None),
])
def various_panel_types(request):
    """Parametrized fixture providing various panel data types."""
    panel_type, groups = request.param
    
    if panel_type == 'simple_panel':
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(20)}, index=idx)
        return X, groups
    elif panel_type == 'unbalanced_panel':
        # Create unbalanced panel
        indices = [
            ('A', pd.Timestamp('2020-01-01')),
            ('A', pd.Timestamp('2020-01-02')),
            ('A', pd.Timestamp('2020-01-03')),
            ('B', pd.Timestamp('2020-01-01')),
            ('B', pd.Timestamp('2020-01-02')),
        ]
        idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(5)}, index=idx)
        return X, groups


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Custom assertions and utilities for tests
class TestHelpers:
    """Helper methods for testing cross-validation functionality."""
    
    @staticmethod
    def assert_valid_split(train_idx, test_idx, X):
        """Assert that a train/test split is valid."""
        # Check types
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        
        # Check bounds
        n_samples = len(X)
        assert np.all(train_idx >= 0)
        assert np.all(train_idx < n_samples)
        assert np.all(test_idx >= 0)
        assert np.all(test_idx < n_samples)
        
        # Check no duplicates within each set
        assert len(np.unique(train_idx)) == len(train_idx)
        assert len(np.unique(test_idx)) == len(test_idx)
    
    @staticmethod
    def assert_temporal_order_preserved(train_idx, test_idx, gap=0):
        """Assert that temporal order is preserved (for out-of-sample)."""
        if len(train_idx) > 0 and len(test_idx) > 0:
            actual_gap = min(test_idx) - max(train_idx) - 1
            assert actual_gap >= gap, f"Gap should be at least {gap}, got {actual_gap}"
    
    @staticmethod
    def assert_insample_property(train_idx, test_idx):
        """Assert in-sample property (test indices included in training)."""
        assert np.all(np.isin(test_idx, train_idx)), "Test indices should be subset of training indices"


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers