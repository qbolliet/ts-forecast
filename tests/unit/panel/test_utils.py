"""Unit tests for ``tsforecast.panel.utils``.

Only ``get_entity_levels``, ``split_variable_key`` and
``normalize_entity_mapping`` are covered here; the
other helpers are exercised through ``PanelwiseTransformer``
(``test_panelwise_transformer.py``).
"""
import pandas as pd
import pytest

from tsforecast.panel import get_entity_levels, is_panel_data
from tsforecast.panel.utils import iter_entity_blocks, normalize_entity_mapping, split_variable_key

DATES = pd.date_range('2023-01-01', periods=3, freq='MS')


class TestGetEntityLevels:
    def test_time_series_has_no_entity_level(self):
        assert get_entity_levels(pd.DataFrame({'x': range(3)}, index=DATES)) == []

    def test_one_entity_level(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], DATES])
        assert get_entity_levels(pd.DataFrame({'x': range(6)}, index=index)) == [0]

    def test_several_entity_levels(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], ['a', 'b'], DATES])
        assert get_entity_levels(pd.Series(range(12), index=index)) == [0, 1]

    def test_series_and_dataframe_agree(self):
        index = pd.MultiIndex.from_product([['FR'], DATES])
        assert get_entity_levels(pd.Series(range(3), index=index)) == get_entity_levels(pd.DataFrame({'x': range(3)}, index=index))

    def test_is_empty_exactly_when_the_data_is_not_a_panel(self):
        for index in (pd.RangeIndex(3), DATES, pd.MultiIndex.from_product([['FR'], DATES])):
            data = pd.Series(range(len(index)), index=index)
            assert bool(get_entity_levels(data)) == is_panel_data(data)

    def test_levels_drop_to_a_date_only_block(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], ['a', 'b'], DATES])
        data = pd.DataFrame({'x': range(12)}, index=index)
        for _, _, block in iter_entity_blocks(data):
            assert block.index.nlevels == data.index.nlevels - len(get_entity_levels(data))


class TestSplitVariableKey:
    """``split_variable_key`` always yields an entity tuple, ``()`` outside a panel."""

    @pytest.mark.parametrize(
        'key, expected',
        [
            pytest.param(('FR', 'gdp'), (('FR',), 'gdp'), id='one_entity_level'),
            pytest.param(('FR', 'manufacturing', 'gdp'), (('FR', 'manufacturing'), 'gdp'),
                         id='two_entity_levels'),
            pytest.param('gdp', ((), 'gdp'), id='time_series_column'),
        ],
    )
    def test_entity_part_is_always_a_tuple(self, key, expected):
        """The entity part is a tuple at every level, the empty tuple for a plain column."""
        assert split_variable_key(key) == expected


class TestNormalizeEntityMapping:
    """``normalize_entity_mapping`` re-keys a per-entity binding on tuples."""

    @pytest.mark.parametrize(
        'mapping, expected',
        [
            pytest.param({'FR': 'Q'}, {('FR',): 'Q'}, id='scalar_key'),
            pytest.param({('FR',): 'Q'}, {('FR',): 'Q'}, id='tuple_key'),
            pytest.param({('Zone euro', 'IT'): 'Y'}, {('Zone euro', 'IT'): 'Y'}, id='two_entity_levels'),
            pytest.param({'FR': 'Q', ('DE',): 'M'}, {('FR',): 'Q', ('DE',): 'M'}, id='mixed_keys'),
            pytest.param({}, {}, id='empty'),
        ],
    )
    def test_keys_become_tuples(self, mapping, expected):
        """Scalar keys become one-element tuples; tuple keys are kept."""
        assert normalize_entity_mapping(mapping) == expected

    def test_order_is_kept_and_input_untouched(self):
        """A new dict, in the order of the input, which stays as it was."""
        mapping = {'IT': 'M', 'FR': 'Y'}
        normalized = normalize_entity_mapping(mapping)
        assert (list(normalized), mapping) == ([('IT',), ('FR',)], {'IT': 'M', 'FR': 'Y'})
