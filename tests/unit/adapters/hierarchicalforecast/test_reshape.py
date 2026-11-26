"""Unit tests of ``tsforecast.adapters.hierarchicalforecast.reshape``.

``stack_nodes`` (time series and panel, one or two column levels, node level just
before the date, empty cells kept, labels and dtypes of the input, errors) and
``unstack_nodes`` (inverse conversion, order of the nodes, missing rows, node level
by name or position, errors), then the wide data of a hierarchy of value columns
reconciled after stacking.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.adapters import stack_nodes, unstack_nodes

# Dates du format large
DATES = pd.date_range('2024-01-01', periods=3, freq='MS', name='date')


def _wide_series() -> pd.DataFrame:
    """Wide time series: one column per node, ``a = b + c``."""
    return pd.DataFrame({'a': [3.0, 5.0, 7.0], 'b': [1.0, 2.0, 3.0], 'c': [2.0, 3.0, 4.0]}, index=DATES)


def _wide_panel() -> pd.DataFrame:
    """Wide panel indexed by ``(entity, date)``."""
    index = pd.MultiIndex.from_product([['A', 'B'], DATES], names=['entity', 'date'])
    return pd.DataFrame({'a': np.arange(6.0) + 10, 'b': np.arange(6.0)}, index=index)


class TestStackNodes:
    """Wide frame to long layout."""

    def test_time_series(self):
        long = stack_nodes(_wide_series())
        assert isinstance(long, pd.Series)
        assert list(long.index.names) == ['node', 'date']
        assert long.index.get_level_values('node').dtype == 'str'
        assert long.loc['c'].tolist() == [2.0, 3.0, 4.0]
        assert long.index.get_level_values('date').unit == DATES.unit
        assert len(long) == 9

    def test_panel_keeps_the_date_last(self):
        long = stack_nodes(_wide_panel(), node_level='component')
        assert list(long.index.names) == ['entity', 'component', 'date']
        assert long.sort_index().loc[('B', 'a')].tolist() == [13.0, 14.0, 15.0]

    def test_two_column_levels_give_one_column_per_variable(self):
        wide = pd.concat({'ets': _wide_series(), 'arima': _wide_series() + 1}, axis=1)
        long = stack_nodes(wide)
        assert isinstance(long, pd.DataFrame)
        assert long.columns.tolist() == ['ets', 'arima']
        assert list(long.index.names) == ['node', 'date']
        np.testing.assert_allclose(long.loc['a', 'arima'], [4.0, 6.0, 8.0])

    def test_empty_cells_are_kept(self):
        wide = _wide_series()
        wide.iloc[1, 1] = np.nan
        long = stack_nodes(wide)
        assert len(long) == 9 and int(long.isna().sum()) == 1

    def test_integer_node_labels_keep_their_dtype(self):
        wide = _wide_series().set_axis([10, 20, 30], axis=1)
        assert stack_nodes(wide).index.get_level_values('node').dtype == np.int64

    def test_input_unchanged(self):
        wide = _wide_series()
        before = wide.copy()
        stack_nodes(wide)
        pd.testing.assert_frame_equal(wide, before)

    @pytest.mark.parametrize(
        ('data', 'params', 'error', 'message'),
        [
            (_wide_series()['a'], {}, TypeError, 'DataFrame'),
            (pd.concat({'x': pd.concat({'m': _wide_series()}, axis=1)}, axis=1), {}, ValueError, 'levels'),
            (_wide_series(), {'node_level': 'date'}, ValueError, 'already names'),
        ],
    )
    def test_errors(self, data, params, error, message):
        with pytest.raises(error, match=message):
            stack_nodes(data, **params)


class TestUnstackNodes:
    """Long layout to wide frame."""

    def test_round_trip_series(self):
        wide = _wide_series()
        pd.testing.assert_frame_equal(unstack_nodes(stack_nodes(wide)), wide.rename_axis(columns='node'))

    def test_round_trip_panel_and_variables(self):
        wide = pd.concat({'ets': _wide_panel(), 'arima': _wide_panel() * 2}, axis=1)
        back = unstack_nodes(stack_nodes(wide))
        assert back.columns.tolist() == [('ets', 'a'), ('ets', 'b'), ('arima', 'a'), ('arima', 'b')]
        pd.testing.assert_frame_equal(back, wide.rename_axis(columns=[None, 'node']))

    def test_nodes_keep_their_order_of_appearance(self):
        wide = _wide_series()[['c', 'a', 'b']]
        assert unstack_nodes(stack_nodes(wide)).columns.tolist() == ['c', 'a', 'b']

    def test_missing_rows_become_missing_values(self):
        long = stack_nodes(_wide_series()).drop(('b', DATES[0]))
        wide = unstack_nodes(long)
        assert np.isnan(wide.loc[DATES[0], 'b']) and wide.loc[DATES[1], 'b'] == 2.0

    def test_node_level_by_position(self):
        long = stack_nodes(_wide_panel())
        assert unstack_nodes(long, node_level=1).columns.tolist() == ['a', 'b']

    @pytest.mark.parametrize(
        ('data', 'params', 'error', 'message'),
        [
            (stack_nodes(_wide_series()), {'node_level': 'x'}, ValueError, 'not an index level'),
            (_wide_series()['a'], {}, ValueError, 'node level and a date level'),
            (stack_nodes(_wide_series()), {'node_level': 4}, ValueError, 'position'),
            ([1.0], {}, TypeError, 'Series or DataFrame'),
        ],
    )
    def test_errors(self, data, params, error, message):
        with pytest.raises(error, match=message):
            unstack_nodes(data, **params)


def test_wide_value_columns_reconciled_after_stacking():
    # Colonnes value_a..value_c d'une hiérarchie, prévisions et réalisations au format large
    pytest.importorskip('hierarchicalforecast')
    from tsforecast.adapters import HierarchicalForecastAdapter

    rename = {'a': 'value_a', 'b': 'value_b', 'c': 'value_c'}
    y_wide = _wide_series().rename(columns=rename)
    X_wide = y_wide + np.array([0.5, -0.2, 0.1])
    y, X = stack_nodes(y_wide), stack_nodes(X_wide).to_frame('model')
    adapter = HierarchicalForecastAdapter({'value_a': ['value_b', 'value_c']}).fit(X, y)
    reconciled = unstack_nodes(pd.Series(adapter.predict(X), index=X.index))
    np.testing.assert_allclose(reconciled['value_a'], reconciled['value_b'] + reconciled['value_c'])
