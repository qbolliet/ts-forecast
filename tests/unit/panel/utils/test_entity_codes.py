"""Unit tests for the integer-code helpers of ``tsforecast.panel.utils``.

Covers ``factorize_row_keys`` (composite keys, missing values, order of first
appearance, length checks) and ``get_entity_codes`` (time series, one- and
two-level panels, unsorted rows, unused levels, agreement with
``get_unique_panel_entities``), with no tuple built per row.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.panel.utils import (
    factorize_row_keys,
    get_entity_codes,
    get_unique_panel_entities,
)


class TestFactorizeRowKeys:
    """``factorize_row_keys`` codes composite row keys in order of first appearance."""

    def test_golden_values_on_two_keys(self):
        """Rows equal on both keys share a code; codes follow the first appearance."""
        codes, first_rows = factorize_row_keys(
            np.array(['FR', 'FR', 'DE', 'FR', 'DE']), np.array(['M', 'Q', 'M', 'M', 'M'])
        )
        assert codes.tolist() == [0, 1, 2, 0, 2]
        assert first_rows.tolist() == [0, 1, 2]

    def test_single_key(self):
        """A single key is a plain factorization."""
        codes, first_rows = factorize_row_keys(pd.Series([3, 1, 3, 2]))
        assert codes.tolist() == [0, 1, 0, 2]
        assert first_rows.tolist() == [0, 1, 3]

    def test_list_key_is_accepted(self):
        """A plain list is accepted as a key, like an array."""
        assert factorize_row_keys(['a', 'b', 'a'])[0].tolist() == [0, 1, 0]

    @pytest.mark.parametrize(
        'key',
        [
            pytest.param(pd.Series([1.0, np.nan, np.nan, 1.0]), id='nan'),
            pytest.param(pd.Series(['x', None, None, 'x'], dtype='str'), id='missing_string'),
            pytest.param(pd.DatetimeIndex(['2023-01-01', 'NaT', 'NaT', '2023-01-01']), id='nat'),
        ],
    )
    def test_missing_value_is_a_value_like_any_other(self, key):
        """The rows holding a missing value share their own code."""
        assert factorize_row_keys(key)[0].tolist() == [0, 1, 1, 0]

    def test_key_combination_does_not_merge_distinct_pairs(self):
        """('a', 'y') and ('b', 'x') never collide, whatever the coding order."""
        first = np.array(['a', 'a', 'b', 'b'])
        second = np.array(['x', 'y', 'x', 'y'])
        assert factorize_row_keys(first, second)[0].tolist() == [0, 1, 2, 3]

    def test_three_keys_of_mixed_types(self):
        """Strings, dates and floats combine into one code per distinct triple."""
        dates = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-01-01'])
        codes, _ = factorize_row_keys(np.array(['a', 'a', 'a', 'a']), dates, pd.Series([1.0, 2.0, 1.0, 1.0]))
        assert codes.tolist() == [0, 1, 2, 0]

    def test_no_key_gives_empty_arrays(self):
        """Without any key there is no row."""
        codes, first_rows = factorize_row_keys()
        assert len(codes) == 0 and len(first_rows) == 0

    def test_empty_keys(self):
        """Keys without rows give empty codes."""
        codes, first_rows = factorize_row_keys(np.array([], dtype=float), np.array([], dtype=float))
        assert codes.tolist() == [] and first_rows.tolist() == []

    def test_keys_of_different_lengths_are_refused(self):
        """Every key must cover the same rows."""
        with pytest.raises(ValueError, match='same length'):
            factorize_row_keys(np.array([1, 2]), np.array([1]))

    def test_codes_are_integers(self):
        """The codes are an integer array, usable as positions."""
        codes, first_rows = factorize_row_keys(np.array(['a', 'b']))
        assert codes.dtype.kind == 'i' and first_rows.dtype.kind == 'i'

    def test_input_is_not_modified(self):
        """The keys are read, never written."""
        key = pd.Series(['b', 'a', 'b'])
        before = key.copy()
        factorize_row_keys(key)
        pd.testing.assert_series_equal(key, before)


class TestGetEntityCodes:
    """``get_entity_codes`` codes the entity of each row from the entity levels."""

    def test_time_series_is_one_degenerate_entity(self, timeseries):
        """A simple index holds the single entity ``()``."""
        codes, entities = get_entity_codes(timeseries.index)
        assert codes.tolist() == [0, 0, 0]
        assert entities == [()]

    def test_two_level_panel(self, panel_2l):
        """One code per entity, keys normalized to one-element tuples."""
        codes, entities = get_entity_codes(panel_2l.index)
        assert codes.tolist() == [0, 0, 0, 1, 1, 1]
        assert entities == [('FR',), ('DE',)]

    def test_three_level_panel(self, panel_3l):
        """The entity is the tuple of all levels but the last."""
        codes, entities = get_entity_codes(panel_3l.index)
        assert codes.tolist() == [0, 0, 1, 1, 2, 2]
        assert entities == [('EU', 'FR'), ('EU', 'DE'), ('AS', 'JP')]

    def test_unsorted_rows_keep_the_order_of_first_appearance(self, panel_2l):
        """Interleaved entities keep their codes and their order of appearance."""
        shuffled = panel_2l.iloc[[3, 0, 4, 1, 5, 2]]
        codes, entities = get_entity_codes(shuffled.index)
        assert codes.tolist() == [0, 1, 0, 1, 0, 1]
        assert entities == [('DE',), ('FR',)]

    def test_unused_levels_are_ignored(self, panel_2l):
        """An entity dropped by a selection has no code."""
        codes, entities = get_entity_codes(panel_2l.iloc[3:].index)
        assert codes.tolist() == [0, 0, 0]
        assert entities == [('DE',)]

    def test_agrees_with_get_unique_panel_entities(self, panel_3l):
        """The entity list is that of ``get_unique_panel_entities``."""
        assert get_entity_codes(panel_3l.index)[1] == get_unique_panel_entities(panel_3l)

    def test_entities_are_python_scalars(self):
        """Integer entity labels come back as Python integers, hashable dictionary keys."""
        index = pd.MultiIndex.from_product([[7, 9], pd.date_range('2023-01-01', periods=2)])
        _, entities = get_entity_codes(index)
        assert entities == [(7,), (9,)]
        assert all(type(entity[0]) is int for entity in entities)

    def test_one_level_multiindex_is_a_time_series(self, dates):
        """A ``MultiIndex`` with a single level holds no entity level."""
        codes, entities = get_entity_codes(pd.MultiIndex.from_arrays([dates]))
        assert codes.tolist() == [0, 0, 0] and entities == [()]

    def test_missing_entity_label_has_its_own_code(self, dates):
        """A missing entity label is a label like any other."""
        index = pd.MultiIndex.from_arrays([['FR', None, None], dates])
        codes, entities = get_entity_codes(index)
        assert codes.tolist() == [0, 1, 1]
        assert entities[0] == ('FR',) and pd.isna(entities[1][0])
