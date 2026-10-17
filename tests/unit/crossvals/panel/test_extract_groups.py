"""Tests of ``_extract_groups_from_panel``, the entity reader of the panel splitters.

The function is private; its public path is ``split`` called without
``groups``, exercised by the other files of this package. Covered here: the
entity of each row for two- and three-level indexes, Series and DataFrame,
integer and tuple entity keys, unbalanced panels, and the refusal of data
that is not a panel.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest

from tsforecast.crossvals.panel import _extract_groups_from_panel

pytestmark = pytest.mark.internal


def _panel(entities: list, periods: int, as_series: bool = False):
    """Balanced daily panel from 2020-01-01, value equal to the row position."""
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    index = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    values = np.arange(len(index))
    return pd.Series(values, index=index) if as_series else pd.DataFrame({'value': values}, index=index)


class TestEntityOfEachRow:
    """One entity label per row, read from every index level but the last."""

    @pytest.mark.parametrize('as_series', [False, True], ids=['frame', 'series'])
    def test_two_levels(self, as_series):
        """With one entity level, each row gets the value of that level."""
        groups = _extract_groups_from_panel(_panel(['A', 'B', 'C'], 3, as_series))
        assert groups.tolist() == ['A'] * 3 + ['B'] * 3 + ['C'] * 3

    def test_integer_entities(self):
        """Integer entity keys are returned unchanged."""
        assert _extract_groups_from_panel(_panel([1, 2], 2)).tolist() == [1, 1, 2, 2]

    def test_several_entity_levels_give_tuples(self):
        """With two entity levels, each group is the (country, sector) tuple."""
        idx = pd.MultiIndex.from_product([['FR'], ['a', 'b'], pd.date_range('2020-01-01', periods=2, freq='D')])
        groups = _extract_groups_from_panel(pd.Series(range(4), index=idx))
        assert groups.tolist() == [('FR', 'a'), ('FR', 'a'), ('FR', 'b'), ('FR', 'b')]

    def test_tuple_keys_in_one_level_stay_whole(self, staggered_panel):
        """A single entity level holding tuples gives one-dimensional tuple labels."""
        keys = {'A': ('FR', 'a'), 'B': ('FR', 'b'), 'C': ('DE', 'a')}
        level = pd.Index([keys[e] for e in staggered_panel.index.get_level_values(0)], tupleize_cols=False)
        X = staggered_panel.copy()
        X.index = pd.MultiIndex.from_arrays([level, staggered_panel.index.get_level_values(1)])
        groups = _extract_groups_from_panel(X)
        assert (groups.shape, groups[0], groups[12]) == ((26,), ('FR', 'a'), ('FR', 'b'))

    def test_unbalanced_panel(self):
        """Entities of different lengths keep one label per row."""
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        idx = pd.MultiIndex.from_tuples([('A', dates[0]), ('A', dates[1]), ('B', dates[0]), ('B', dates[1]),
                                         ('B', dates[2])], names=['entity', 'date'])
        groups = _extract_groups_from_panel(pd.DataFrame({'value': range(5)}, index=idx))
        assert groups.tolist() == ['A', 'A', 'B', 'B', 'B']

    def test_rows_keep_their_order(self):
        """Labels follow the row order, interleaved entities included."""
        X = _panel(['A', 'B'], 2).swaplevel().sort_index().swaplevel()
        assert _extract_groups_from_panel(X).tolist() == ['A', 'B', 'A', 'B']


class TestRefusals:
    """Data without a MultiIndex (entity..., date) is not a panel."""

    def test_array_is_refused(self):
        """An array has no index to read the entities from."""
        with pytest.raises(ValueError, match='Panel data requires pandas DataFrame/Series with MultiIndex'):
            _extract_groups_from_panel(np.arange(10))

    @pytest.mark.parametrize(
        'X',
        [
            pytest.param(pd.DataFrame({'value': range(4)}), id='frame'),
            pytest.param(pd.Series(range(4), index=pd.date_range('2020-01-01', periods=4)), id='series'),
        ],
    )
    def test_flat_index_is_refused(self, X):
        """A flat index is not panel data."""
        with pytest.raises(ValueError, match=r'Panel data requires MultiIndex \(entity, time\)'):
            _extract_groups_from_panel(X)
