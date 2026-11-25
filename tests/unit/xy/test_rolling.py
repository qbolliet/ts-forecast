"""Unit tests of ``tsforecast.xy.rolling``.

Covers ``RollingAggregator``: golden values of past and future windows
(period-end and period-start grids, quarterly, yearly and multiplied
windows, semi-monthly and daily grids), the two passes (``fit_transform``
aggregates ``y``, ``transform`` gives it back unchanged; Series, DataFrame
and array targets, misaligned targets), incomplete windows (edges, NaN,
missing grid dates), panels (no window across entities, heterogeneous grids,
unknown entities, unsorted rows, entities in columns), per-column
aggregations and ``suffix``, parameter and grid refusals, date resolution,
non-mutation of the inputs and the sklearn contract.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.xy import DropNaNTarget, RollingAggregator, XYPipeline, is_xy_transformer
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution


# =============================================================================
# Données
# =============================================================================
def _series(freq: str, values, start: str = '2024-01-01', name: str = 'x') -> pd.DataFrame:
    """One-column frame on a regular grid."""
    dates = pd.date_range(start, periods=len(values), freq=freq, name='date')
    return pd.DataFrame({name: np.asarray(values, dtype=float)}, index=dates)


def _panel(freq: str = 'ME', n: int = 6) -> pd.DataFrame:
    """Two entities on the same grid, values 1..n for A and 10..10n for B."""
    dates = pd.date_range('2024-01-01', periods=n, freq=freq)
    index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    values = np.concatenate([np.arange(1, n + 1), 10 * np.arange(1, n + 1)]).astype(float)
    return pd.DataFrame({'x': values}, index=index)


ONE_TO_SIX = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# =============================================================================
# Valeurs d'or
# =============================================================================
class TestGoldenValues:
    """Window direction, length and aggregations on small grids."""

    def test_period_end_grid_gives_past_window(self):
        out = RollingAggregator(window='Q').fit_transform(_series('ME', ONE_TO_SIX))
        assert out['x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]
        assert out['x'].iloc[:2].isna().all()

    def test_period_start_grid_gives_future_window(self):
        out = RollingAggregator(window='Q').fit_transform(_series('MS', ONE_TO_SIX))
        assert out['x'].tolist()[:4] == [6.0, 9.0, 12.0, 15.0]
        assert out['x'].iloc[4:].isna().all()

    def test_anchor_rows_sum_their_own_quarter(self):
        # Mars (fin de trimestre, grille ME) et janvier (début, grille MS) : le trimestre civil
        end = RollingAggregator(window='QE').fit_transform(_series('ME', ONE_TO_SIX))
        start = RollingAggregator(window='QS').fit_transform(_series('MS', ONE_TO_SIX))
        assert end.loc['2024-03-31', 'x'] == 6.0 and end.loc['2024-06-30', 'x'] == 15.0
        assert start.loc['2024-01-01', 'x'] == 6.0 and start.loc['2024-04-01', 'x'] == 15.0

    def test_yearly_window_on_monthly_grid_has_twelve_steps(self):
        step = RollingAggregator(window='Y')
        out = step.fit_transform(_series('ME', np.ones(14)))
        assert step.window_lengths_ == {(): 12}
        assert out['x'].iloc[11:].tolist() == [12.0, 12.0, 12.0]
        assert out['x'].iloc[:11].isna().all()

    def test_yearly_window_on_quarterly_grid_has_four_steps(self):
        step = RollingAggregator(window='Y')
        out = step.fit_transform(_series('QE', [1.0, 2.0, 3.0, 4.0, 5.0]))
        assert step.window_lengths_ == {(): 4}
        assert out['x'].tolist()[3:] == [10.0, 14.0]

    def test_multiplied_window_of_the_grid_frequency(self):
        step = RollingAggregator(window='2M')
        out = step.fit_transform(_series('ME', ONE_TO_SIX))
        assert step.window_lengths_ == {(): 2}
        assert out['x'].tolist()[1:] == [3.0, 5.0, 7.0, 9.0, 11.0]

    def test_window_equal_to_grid_is_identity(self):
        X = _series('QE', [1.0, 2.0, 3.0])
        assert RollingAggregator(window='Q').fit_transform(X)['x'].tolist() == [1.0, 2.0, 3.0]

    def test_explicit_alignment_overrides_auto(self):
        step = RollingAggregator(window='Q', alignment='future')
        out = step.fit_transform(_series('ME', ONE_TO_SIX))
        assert step.alignments_ == {(): 'future'}
        assert out['x'].tolist()[:4] == [6.0, 9.0, 12.0, 15.0]

    @pytest.mark.parametrize('agg, expected', [
        ('sum', [6.0, 9.0, 12.0, 15.0]),
        ('mean', [2.0, 3.0, 4.0, 5.0]),
        ('min', [1.0, 2.0, 3.0, 4.0]),
        ('max', [3.0, 4.0, 5.0, 6.0]),
        ('first', [1.0, 2.0, 3.0, 4.0]),
        ('last', [3.0, 4.0, 5.0, 6.0]),
    ])
    def test_aggregations(self, agg, expected):
        out = RollingAggregator(window='Q', agg=agg).fit_transform(_series('ME', ONE_TO_SIX))
        assert out['x'].tolist()[2:] == expected

    @pytest.mark.parametrize('freq, alignment, expected', [
        ('SME', 'past', [np.nan, 3.0, 5.0, 7.0]),
        ('SMS', 'future', [3.0, 5.0, 7.0, np.nan]),
    ])
    def test_semi_monthly_grid(self, freq, alignment, expected):
        step = RollingAggregator(window='M')
        out = step.fit_transform(_series(freq, [1.0, 2.0, 3.0, 4.0]))
        assert step.window_lengths_ == {(): 2} and step.alignments_ == {(): alignment}
        np.testing.assert_array_equal(out['x'].to_numpy(), expected)

    def test_weekly_window_on_daily_grid_with_explicit_alignment(self):
        out = RollingAggregator(window='W', alignment='past').fit_transform(_series('D', np.ones(8)))
        assert out['x'].tolist()[6:] == [7.0, 7.0]

    def test_fitted_attributes(self):
        step = RollingAggregator(window='Q', agg={'x': 'mean'}).fit(_series('MS', ONE_TO_SIX))
        assert step.grid_frequencies_ == {(): 'MS'}
        assert step.window_lengths_ == {(): 3}
        assert step.alignments_ == {(): 'future'}
        assert step.aggregations_ == {'x': 'mean'}


# =============================================================================
# Passes d'entraînement et de prédiction
# =============================================================================
class TestPasses:
    """``y`` aggregated at the training pass only."""

    @pytest.fixture
    def xy(self):
        X = _series('ME', ONE_TO_SIX)
        y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=X.index, name='target')
        return X, y

    def test_fit_transform_aggregates_x_and_y(self, xy):
        X, y = xy
        X_t, y_t = RollingAggregator(window='Q').fit_transform(X, y)
        assert X_t['x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]
        assert y_t.tolist()[2:] == [60.0, 90.0, 120.0, 150.0]
        assert y_t.name == 'target' and y_t.index.equals(X.index) and y_t.dtype == 'float64'

    def test_transform_gives_y_back_unchanged(self, xy):
        X, y = xy
        step = RollingAggregator(window='Q').fit(X, y)
        X_t, y_t = step.transform(X, y)
        assert y_t is y
        assert X_t['x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]

    def test_transform_without_y_returns_x_only(self, xy):
        X, y = xy
        out = RollingAggregator(window='Q').fit(X, y).transform(X)
        assert isinstance(out, pd.DataFrame)

    def test_fit_transform_without_y_returns_x_only(self, xy):
        X, _ = xy
        assert isinstance(RollingAggregator(window='Q').fit_transform(X), pd.DataFrame)

    def test_target_agg(self, xy):
        X, y = xy
        _, y_t = RollingAggregator(window='Q', agg='sum', target_agg='mean').fit_transform(X, y)
        assert y_t.tolist()[2:] == [20.0, 30.0, 40.0, 50.0]

    def test_multi_output_target(self, xy):
        X, y = xy
        Y = pd.DataFrame({'a': y, 'b': 2 * y})
        _, Y_t = RollingAggregator(window='Q').fit_transform(X, Y)
        assert list(Y_t.columns) == ['a', 'b']
        assert Y_t['b'].tolist()[2:] == [120.0, 180.0, 240.0, 300.0]

    def test_array_target_matched_by_position(self, xy):
        X, y = xy
        _, y_t = RollingAggregator(window='Q').fit_transform(X, y.to_numpy())
        assert isinstance(y_t, np.ndarray) and y_t.shape == (6,)
        np.testing.assert_array_equal(y_t[2:], [60.0, 90.0, 120.0, 150.0])

    def test_shuffled_target_is_realigned_on_x(self, xy):
        X, y = xy
        _, y_t = RollingAggregator(window='Q').fit_transform(X, y.iloc[::-1])
        assert y_t.index.equals(X.index)
        assert y_t.tolist()[2:] == [60.0, 90.0, 120.0, 150.0]

    def test_target_on_other_rows_is_refused(self, xy):
        X, y = xy
        with pytest.raises(ValueError, match='rows of X'):
            RollingAggregator(window='Q').fit_transform(X, y.iloc[:-1])

    def test_array_target_of_other_length_is_refused(self, xy):
        X, y = xy
        with pytest.raises(ValueError, match='same length'):
            RollingAggregator(window='Q').fit_transform(X, y.to_numpy()[:-1])

    def test_is_an_xy_step_of_the_pipeline(self, xy):
        X, y = xy
        assert is_xy_transformer(RollingAggregator(window='Q'))
        pipe = XYPipeline([
            ('roll', RollingAggregator(window='Q')),
            ('drop', DropNaNTarget()),
            ('model', LinearRegression()),
        ])
        pipe.fit(X, y)
        # Entraînement sur les quatre trimestres glissants complets : y = 10 x
        np.testing.assert_allclose(pipe[-1].coef_, [10.0])
        X_t, y_t = pipe[:-1].transform(X, y)
        assert len(X_t) == len(y_t)


# =============================================================================
# Fenêtres incomplètes
# =============================================================================
class TestIncompleteWindows:
    """A window missing a value gives NaN."""

    def test_nan_value_spoils_every_window_reading_it(self):
        out = RollingAggregator(window='Q').fit_transform(_series('ME', [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]))
        assert out['x'].iloc[:5].isna().all()
        assert out['x'].iloc[5] == 15.0

    def test_missing_grid_date_is_a_hole_not_a_shift(self):
        X = _series('ME', ONE_TO_SIX).drop(pd.Timestamp('2024-03-31'))
        out = RollingAggregator(window='Q').fit_transform(X)
        # Mai lirait mars : fenêtre incomplète ; juin ne lit que avril-juin
        assert out['x'].isna().tolist() == [True, True, True, True, False]
        assert out['x'].iloc[-1] == 15.0

    def test_two_rows_in_one_grid_period_are_refused(self):
        X = _series('ME', ONE_TO_SIX)
        step = RollingAggregator(window='Q').fit(X)
        other = pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(
            ['2024-01-15', '2024-01-31', '2024-02-29'], name='date'
        ))
        with pytest.raises(ValueError, match='same'):
            step.transform(other)

    def test_short_test_set_has_nan_edges(self):
        step = RollingAggregator(window='Q').fit(_series('ME', ONE_TO_SIX))
        test = _series('ME', [7.0, 8.0, 9.0], start='2024-07-01')
        assert step.transform(test)['x'].isna().tolist() == [True, True, False]

    def test_empty_transform_input(self):
        step = RollingAggregator(window='Q').fit(_series('ME', ONE_TO_SIX))
        empty = _series('ME', ONE_TO_SIX).iloc[:0]
        out = step.transform(empty)
        assert out.empty and list(out.columns) == ['x']


# =============================================================================
# Panels
# =============================================================================
class TestPanel:
    """Windows computed entity by entity."""

    def test_windows_never_cross_entities(self):
        out = RollingAggregator(window='Q').fit_transform(_panel())
        assert out.loc['A', 'x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]
        assert out.loc['B', 'x'].tolist()[2:] == [60.0, 90.0, 120.0, 150.0]
        assert out.loc['B', 'x'].iloc[:2].isna().all()

    def test_heterogeneous_grids_per_entity(self):
        monthly = pd.date_range('2024-01-31', periods=6, freq='ME')
        quarterly = pd.date_range('2024-03-31', periods=4, freq='QE')
        index = pd.MultiIndex.from_tuples(
            [('A', d) for d in monthly] + [('B', d) for d in quarterly], names=['entity', 'date']
        )
        X = pd.DataFrame({'x': ONE_TO_SIX + [1.0, 2.0, 3.0, 4.0]}, index=index)
        step = RollingAggregator(window='Q')
        out = step.fit_transform(X)
        assert step.window_lengths_ == {('A',): 3, ('B',): 1}
        assert step.grid_frequencies_ == {('A',): 'ME', ('B',): 'QE'}
        assert out.loc['B', 'x'].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_unknown_entity_takes_the_shared_window(self):
        step = RollingAggregator(window='Q').fit(_panel())
        dates = pd.date_range('2024-01-01', periods=3, freq='ME')
        new = pd.DataFrame(
            {'x': [1.0, 1.0, 1.0]},
            index=pd.MultiIndex.from_product([['C'], dates], names=['entity', 'date']),
        )
        assert step.transform(new)['x'].tolist()[2] == 3.0

    def test_unknown_entity_without_shared_window_reads_its_own_grid(self):
        monthly = pd.date_range('2024-01-31', periods=3, freq='ME')
        quarterly = pd.date_range('2024-03-31', periods=2, freq='QE')
        index = pd.MultiIndex.from_tuples(
            [('A', d) for d in monthly] + [('B', d) for d in quarterly], names=['entity', 'date']
        )
        step = RollingAggregator(window='Q').fit(pd.DataFrame({'x': np.ones(5)}, index=index))
        new_dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        new = pd.DataFrame(
            {'x': np.ones(4)},
            index=pd.MultiIndex.from_product([['C'], new_dates], names=['entity', 'date']),
        )
        assert step.transform(new)['x'].tolist()[:2] == [3.0, 3.0]

    def test_unsorted_rows_with_auto_sort(self):
        X = _panel()
        shuffled = X.iloc[[3, 0, 9, 1, 5, 2, 4, 11, 6, 8, 7, 10]]
        out = RollingAggregator(window='Q', auto_sort=True).fit_transform(shuffled)
        expected = RollingAggregator(window='Q').fit_transform(X)
        pd.testing.assert_frame_equal(out.sort_index(), expected.sort_index())

    def test_entities_in_columns(self):
        X = _panel().reset_index()
        step = RollingAggregator(window='Q', time_col='date', panel_cols=['entity'])
        out = step.fit_transform(X)
        assert list(out.columns) == ['entity', 'date', 'x']
        assert out['entity'].equals(X['entity']) and out['date'].equals(X['date'])
        assert out.loc[out['entity'] == 'B', 'x'].tolist()[2:] == [60.0, 90.0, 120.0, 150.0]
        assert step.get_feature_names_out().tolist() == ['entity', 'date', 'x']

    def test_time_series_with_time_column(self):
        X = _series('ME', ONE_TO_SIX).reset_index()
        out = RollingAggregator(window='Q', time_col='date').fit_transform(X)
        assert out['x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]

    def test_panel_target_is_aggregated_per_entity(self):
        X = _panel()
        y = X['x'].rename('y')
        _, y_t = RollingAggregator(window='Q').fit_transform(X, y)
        assert y_t.loc['B'].tolist()[2:] == [60.0, 90.0, 120.0, 150.0]


# =============================================================================
# Colonnes et agrégations par colonne
# =============================================================================
class TestColumns:
    """Per-column aggregations, ``suffix`` and output names."""

    @pytest.fixture
    def two_columns(self):
        X = _series('ME', ONE_TO_SIX)
        X['rate'] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        return X

    def test_per_column_aggregation_with_default(self, two_columns):
        step = RollingAggregator(window='Q', agg={'rate': 'mean', '__default__': 'max'})
        out = step.fit_transform(two_columns)
        assert out['x'].tolist()[2:] == [3.0, 4.0, 5.0, 6.0]
        assert out['rate'].tolist()[2:] == [4.0, 6.0, 8.0, 10.0]

    def test_per_column_aggregation_defaults_to_sum(self, two_columns):
        out = RollingAggregator(window='Q', agg={'rate': 'mean'}).fit_transform(two_columns)
        assert out['x'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]

    def test_unknown_keys_warn_once(self, two_columns):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RollingAggregator(window='Q', agg={'typo': 'mean', 'other': 'sum'}).fit(two_columns)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert len(messages) == 1 and 'typo' in messages[0] and 'other' in messages[0]

    def test_suffix_keeps_sources_and_appends_aggregates(self, two_columns):
        step = RollingAggregator(window='Q', suffix='_q')
        out = step.fit_transform(two_columns)
        assert list(out.columns) == ['x', 'rate', 'x_q', 'rate_q']
        assert out['x'].tolist() == ONE_TO_SIX
        assert out['x_q'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]
        assert step.get_feature_names_out().tolist() == ['x', 'rate', 'x_q', 'rate_q']

    def test_suffix_colliding_with_a_column_is_refused(self, two_columns):
        X = two_columns.rename(columns={'rate': 'x_q'})
        with pytest.raises(ValueError, match='already'):
            RollingAggregator(window='Q', suffix='_q').fit(X)

    def test_integer_and_boolean_columns_come_back_as_float(self):
        X = _series('ME', ONE_TO_SIX)
        X['n'] = np.arange(6)
        X['flag'] = [True, False, True, True, False, True]
        out = RollingAggregator(window='Q').fit_transform(X)
        assert (out.dtypes == 'float64').all()
        assert out['flag'].tolist()[2:] == [2.0, 2.0, 2.0, 2.0]

    def test_special_column_names(self):
        X = _series('ME', ONE_TO_SIX, name='PIB (en €) / hab.')
        out = RollingAggregator(window='Q', suffix=' [Q]').fit_transform(X)
        assert out['PIB (en €) / hab. [Q]'].tolist()[2:] == [6.0, 9.0, 12.0, 15.0]

    def test_feature_names_are_checked_at_transform(self, two_columns):
        step = RollingAggregator(window='Q').fit(two_columns)
        with pytest.raises(ValueError):
            step.transform(two_columns[['rate', 'x']])


# =============================================================================
# Refus
# =============================================================================
class TestRefusals:
    """Invalid parameters and grids."""

    @pytest.mark.parametrize('params, match', [
        ({'window': 3}, 'window'),
        ({'window': 'bad'}, 'window'),
        ({'window': 'Q', 'agg': 'median'}, 'agg'),
        ({'window': 'Q', 'agg': {'x': 'median'}}, 'agg'),
        ({'window': 'Q', 'target_agg': 'median'}, 'target_agg'),
        ({'window': 'Q', 'alignment': 'centre'}, 'alignment'),
        ({'window': 'Q', 'suffix': ''}, 'suffix'),
    ])
    def test_invalid_parameters(self, params, match):
        with pytest.raises(ValueError, match=match):
            RollingAggregator(**params).fit(_series('ME', ONE_TO_SIX))

    def test_window_finer_than_grid(self):
        with pytest.raises(ValueError, match='exact number'):
            RollingAggregator(window='M').fit(_series('QE', [1.0, 2.0, 3.0]))

    def test_window_without_exact_number_of_grid_periods(self):
        with pytest.raises(ValueError, match='exact number'):
            RollingAggregator(window='Q').fit(_series('D', np.ones(10)))

    def test_auto_alignment_on_grid_without_position(self):
        with pytest.raises(ValueError, match='alignment'):
            RollingAggregator(window='W').fit(_series('D', np.ones(10)))

    def test_undetectable_grid(self):
        with pytest.raises(ValueError, match='cannot be detected'):
            RollingAggregator(window='Q').fit(_series('ME', [1.0]))

    def test_grid_with_multiplier(self):
        with pytest.raises(ValueError, match='multiplier'):
            RollingAggregator(window='Y').fit(_series('2ME', ONE_TO_SIX))

    def test_stacked_frequency_level(self):
        X = _series('ME', ONE_TO_SIX)
        stacked = pd.concat({'M': X}, names=['frequency'])
        with pytest.raises(ValueError, match='keep_lower_frequencies'):
            RollingAggregator(window='Q').fit(stacked)

    def test_non_numeric_column(self):
        X = _series('ME', ONE_TO_SIX)
        X['label'] = pd.Series(list('abcdef'), index=X.index, dtype='str')
        with pytest.raises(TypeError, match='label'):
            RollingAggregator(window='Q').fit(X)

    def test_inverse_transform_is_not_implemented(self):
        X = _series('ME', ONE_TO_SIX)
        step = RollingAggregator(window='Q').fit(X)
        with pytest.raises(NotImplementedError):
            step.inverse_transform(X)


# =============================================================================
# Robustesse
# =============================================================================
class TestDatetimeResolution:
    """Same instants at every storage unit: same values, input unit kept."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_result_independent_of_unit(self, unit):
        X = with_datetime_resolution(_panel(), unit)
        y = with_datetime_resolution(_panel()['x'].rename('y'), unit)
        X_t, y_t = RollingAggregator(window='Q').fit_transform(X, y)
        reference = RollingAggregator(window='Q').fit_transform(_panel())
        np.testing.assert_array_equal(X_t['x'].to_numpy(), reference['x'].to_numpy())
        assert datetime_units(X_t) == {('index', 1): unit}
        assert datetime_units(y_t) == {('index', 1): unit}


class TestInputContract:
    """Inputs never modified; sklearn contract."""

    def test_inputs_are_not_modified(self):
        X = _panel()
        y = X['x'].rename('y')
        X_copy, y_copy = X.copy(), y.copy()
        RollingAggregator(window='Q', suffix='_q').fit_transform(X, y)
        pd.testing.assert_frame_equal(X, X_copy)
        pd.testing.assert_series_equal(y, y_copy)

    def test_read_only_input(self):
        X = _series('ME', ONE_TO_SIX)
        values = X['x'].to_numpy()
        assert not values.flags.writeable
        assert RollingAggregator(window='Q').fit_transform(X)['x'].iloc[-1] == 15.0

    def test_clone_and_params(self):
        step = RollingAggregator(window='Y', agg={'x': 'mean'}, alignment='past', suffix='_y')
        params = clone(step).get_params()
        assert params['window'] == 'Y' and params['agg'] == {'x': 'mean'}
        assert params['alignment'] == 'past' and params['suffix'] == '_y'

    def test_set_output_is_accepted(self):
        step = RollingAggregator(window='Q')
        assert step.set_output(transform='pandas') is step
        assert isinstance(step.fit_transform(_series('ME', ONE_TO_SIX)), pd.DataFrame)

    def test_period_index_is_read_as_dates(self):
        X = _series('ME', ONE_TO_SIX)
        periods = X.set_axis(X.index.to_period('M'))
        out = RollingAggregator(window='Q').fit_transform(periods)
        assert out['x'].tolist()[:4] == [6.0, 9.0, 12.0, 15.0]
