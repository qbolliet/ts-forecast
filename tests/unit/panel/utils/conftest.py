"""Shared fixtures of the ``tsforecast.panel.utils`` tests: small hand-built panels.

Every panel has values that can be read off by hand, so that the golden values
of the tests are written before running the code under test.
"""
import pandas as pd
import pytest


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    """Three month-start dates: 2023-01-01, 2023-02-01, 2023-03-01."""
    return pd.date_range('2023-01-01', periods=3, freq='MS')


@pytest.fixture
def panel_2l(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Two-level panel (entity, date): FR holds 1, 2, 3 and DE holds 4, 5, 6 in column ``x``."""
    index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['entity', 'date'])
    return pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]}, index=index)


@pytest.fixture
def panel_3l(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Three-level panel (region, country, date) of three entities with two dates each.

    Entities in order of appearance: ``('EU', 'FR')`` holds 0, 1, ``('EU', 'DE')``
    holds 2, 3 and ``('AS', 'JP')`` holds 4, 5 in column ``x``.
    """
    keys = [('EU', 'FR'), ('EU', 'DE'), ('AS', 'JP')]
    index = pd.MultiIndex.from_tuples(
        [(*key, date) for key in keys for date in dates[:2]],
        names=['region', 'country', 'date'],
    )
    return pd.DataFrame({'x': range(6)}, index=index)


@pytest.fixture
def timeseries(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Time series (``DatetimeIndex``) holding 1, 2, 3 in column ``x``."""
    return pd.DataFrame({'x': [1, 2, 3]}, index=dates)
