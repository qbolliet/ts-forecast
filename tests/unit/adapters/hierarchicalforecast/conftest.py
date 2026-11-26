"""Shared fixtures of the ``tsforecast.adapters.hierarchicalforecast`` unit tests.

Hierarchical datasets in long layout split on the dates (the last three dates
for test): the node tree ``a = b + c``, ``c = d + e`` as a time series
(``tree_series``) and as a panel of independent copies (``tree_panel``), and the
country / region hierarchy of index levels (``levels_series``, ``levels_panel``).
"""
import pytest

from tests.support.adapter_contract import split_by_date
from tests.support.datasets import build_hierarchy_levels, build_hierarchy_nodes


@pytest.fixture
def tree_series():
    """``(X_train, y_train, X_test, y_test)`` of the node tree, index ``(node, date)``."""
    return split_by_date(*build_hierarchy_nodes('series'))


@pytest.fixture
def tree_panel():
    """``(X_train, y_train, X_test, y_test)`` of the node tree copied on entities ``A``, ``B``, ``C``."""
    return split_by_date(*build_hierarchy_nodes('panel'))


@pytest.fixture
def levels_series():
    """``(X_train, y_train, X_test, y_test)`` of the hierarchy ``(country, region, date)``."""
    return split_by_date(*build_hierarchy_levels('series'))


@pytest.fixture
def levels_panel():
    """``(X_train, y_train, X_test, y_test)`` of the hierarchy copied on markets ``A``, ``B``, ``C``."""
    return split_by_date(*build_hierarchy_levels('panel'))
