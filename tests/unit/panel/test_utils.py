"""Unit tests for ``tsforecast.panel.utils``.

Only ``get_entity_levels`` is covered here; the other helpers are exercised
through ``PanelwiseTransformer`` (``test_panelwise_transformer.py``).
"""
import pandas as pd
import pytest

from tsforecast.panel import get_entity_levels, is_panel_data
from tsforecast.panel.utils import iter_entity_blocks

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
