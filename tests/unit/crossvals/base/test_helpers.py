"""Tests of the private helpers of ``tsforecast/crossvals/base.py`` without a public path.

Covers ``_precompute_group_mappings`` (lookup table with no observable effect
of its own) and the permutation returned by ``_verify_and_sort_data`` for
unsorted data, which the public folds cannot show independently of the
remapping that uses it. Every other behaviour of the helpers is tested
through ``split`` in the neighbouring files.
"""
# Importation des modules
import warnings

import numpy as np
import pandas as pd
import pytest

from tsforecast.crossvals.base import _precompute_group_mappings, _verify_and_sort_data

pytestmark = pytest.mark.internal


class TestPrecomputeGroupMappings:
    """Positions of the rows of each group."""

    @pytest.mark.parametrize(
        ('groups', 'expected'),
        [
            (np.array(['A', 'A', 'B', 'B', 'A']), {'A': [0, 1, 4], 'B': [2, 3]}),
            (np.array(['A', 'A', 'A']), {'A': [0, 1, 2]}),
            (np.array([1, 2, 1, 3, 2]), {1: [0, 2], 2: [1, 4], 3: [3]}),
            (np.array([]), {}),
        ],
        ids=['interleaved', 'single-group', 'numeric', 'empty'],
    )
    def test_positions_per_group(self, groups, expected):
        """Each group maps to the sorted positions of its rows."""
        result = _precompute_group_mappings(groups)
        assert {key: value.tolist() for key, value in result.items()} == expected


class TestSortPermutation:
    """Permutation from sorted rows to original rows."""

    def test_time_series(self):
        """sort_indices[k] is the original position of the k-th date."""
        # Construction : la ligne i porte la date numéro perm[i]
        perm = [2, 0, 4, 1, 3]
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        X = pd.Series(perm, index=dates[perm])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X)
        # Valeur d'or : la date 0 est en ligne 1, la date 1 en ligne 3, etc.
        assert sort_indices.tolist() == [1, 3, 0, 4, 2]
        assert X_sorted.index.equals(dates) and groups_sorted is None

    def test_panel(self):
        """Rows are ordered by entity, then by date within each entity."""
        entities = ['B', 'A', 'B', 'A']
        dates = pd.to_datetime(['2020-01-02', '2020-01-01', '2020-01-01', '2020-01-02'])
        X = pd.DataFrame({'value': [0, 1, 2, 3]},
                         index=pd.MultiIndex.from_arrays([entities, dates], names=['entity', 'date']))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            _, groups_sorted, sort_indices = _verify_and_sort_data(X, np.array(entities))
        # Valeur d'or : (A, 01) ligne 1, (A, 02) ligne 3, (B, 01) ligne 2, (B, 02) ligne 0
        assert sort_indices.tolist() == [1, 3, 2, 0]
        assert groups_sorted.tolist() == ['A', 'A', 'B', 'B']
