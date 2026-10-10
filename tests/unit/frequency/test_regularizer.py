"""Tests for tsforecast.frequency.regularizer.

``IndexRegularizer`` is exported but not used by ``HighFrequencyImputer``; its
contract is therefore read from its own docstrings and from the finality of the
component: detect an irregular temporal index and fill its gaps with NaN rows,
**without ever destroying an observation or changing the bounds of a series**.

Covered symbols: ``is_regular`` and ``regularize`` (module functions),
``IndexRegularizer.is_regular`` and ``IndexRegularizer.regularize`` (methods),
with ``time_col`` / ``panel_cols`` / ``per_entity``, on handmade series and
panels (golden values computed by hand) and on the realistic datasets
``irregular_index_timeseries`` and ``heterogeneous_coverage_panel``.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.regularizer import (
    IndexRegularizer,
    is_regular,
    regularize,
)

# Les sept fréquences du périmètre : jour, mois (début / fin), trimestre, année
_FREQUENCIES = ['D', 'MS', 'ME', 'QS', 'QE', 'YS', 'YE']


def _monthly(periods: int = 6, start: str = '2020-01-01') -> pd.Series:
    """Build a regular month-start series of floats ``0.0 .. periods - 1``."""
    index = pd.date_range(start, periods=periods, freq='MS')
    return pd.Series(np.arange(periods, dtype=float), index=index)


def _panel(**entities: pd.Series) -> pd.Series:
    """Stack one series per entity into a (entity, date) panel."""
    panel = pd.concat(entities)
    panel.index.names = ['entity', 'date']
    return panel


# ------------------------------------------------------------------ #
#  is_regular                                                         #
# ------------------------------------------------------------------ #

class TestIsRegularTimeSeries:
    """Regularity of a single DatetimeIndex."""

    @pytest.mark.parametrize('freq', _FREQUENCIES)
    def test_complete_grid_is_regular(self, freq):
        """A gap-free grid is regular for every supported frequency."""
        # Construction d'une grille complète de huit dates
        series = pd.Series(range(8), index=pd.date_range('2020-01-01', periods=8, freq=freq))
        assert is_regular(series) is True

    @pytest.mark.parametrize('freq', _FREQUENCIES)
    def test_grid_with_a_gap_is_irregular(self, freq):
        """Dropping one interior date makes the index irregular."""
        # Suppression de la quatrième date : trou interne
        series = pd.Series(range(8), index=pd.date_range('2020-01-01', periods=8, freq=freq))
        assert is_regular(series.drop(series.index[3])) is False

    def test_duplicated_timestamp_is_irregular(self):
        """A duplicated timestamp breaks regularity."""
        dates = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-02-01', '2020-03-01'])
        assert is_regular(pd.Series(range(4), index=dates)) is False

    @pytest.mark.parametrize('n_obs', [0, 1, 2])
    def test_fewer_than_three_dates_cannot_be_regular(self, n_obs):
        """Pandas needs three dates to infer a frequency: shorter indexes are not regular."""
        assert is_regular(_monthly(6).iloc[:n_obs]) is False

    def test_dataframe_is_accepted(self):
        """A DataFrame is checked on its index like a Series."""
        frame = _monthly().to_frame('x')
        assert is_regular(frame) is True

    def test_time_col_is_used_as_index(self):
        """``time_col`` designates the column holding the timestamps."""
        # Colonne de dates avec un trou en mars
        frame = pd.DataFrame({'date': _monthly().index.delete(2), 'x': range(5)})
        assert is_regular(frame, time_col='date') is False

    def test_irregular_index_timeseries_is_irregular(self, irregular_index_timeseries):
        """Isolated annual anchors before the monthly grid make the index irregular."""
        assert is_regular(irregular_index_timeseries) is False

    def test_unsorted_but_complete_index_is_regular(self):
        """Row order is not a regularity criterion: a shuffled gap-free index is regular."""
        series = _monthly()
        assert is_regular(series.iloc[[3, 0, 5, 1, 4, 2]]) is True

    def test_unsorted_index_with_a_gap_is_irregular(self):
        """Sorting does not hide a gap."""
        series = _monthly()
        assert is_regular(series.drop(series.index[2]).iloc[[3, 0, 4, 1, 2]]) is False

    def test_integer_index_raises(self):
        """Integer labels are not dates: the temporal index is rejected."""
        with pytest.raises(TypeError, match="must be a DatetimeIndex or a PeriodIndex"):
            is_regular(pd.Series(range(4), index=pd.RangeIndex(4)))

    @pytest.mark.parametrize('freq', ['D', 'M', 'Q', 'Y'])
    def test_complete_period_index_is_regular(self, freq):
        """A gap-free PeriodIndex is regular."""
        series = pd.Series(range(8), index=pd.period_range('2020-01-01', periods=8, freq=freq))
        assert is_regular(series) is True

    @pytest.mark.parametrize('freq', ['D', 'M', 'Q', 'Y'])
    def test_period_index_with_a_gap_is_irregular(self, freq):
        """A hole in a PeriodIndex is detected."""
        series = pd.Series(range(8), index=pd.period_range('2020-01-01', periods=8, freq=freq))
        assert is_regular(series.drop(series.index[3])) is False

    def test_period_panel_is_regular_per_entity(self):
        """The last level of a panel may be a PeriodIndex."""
        periods = pd.period_range('2020-01', periods=6, freq='M')
        panel = pd.concat({'A': pd.Series(range(6), index=periods),
                           'B': pd.Series(range(5), index=periods.delete(2))},
                          names=['entity', 'date'])
        assert is_regular(panel, per_entity=True) == {('A',): True, ('B',): False}


class TestIsRegularPanel:
    """Regularity of panel data, global and per entity."""

    def test_regular_panel_is_regular(self):
        """Every entity regular with the same frequency gives True."""
        panel = _panel(A=_monthly(), B=_monthly(5))
        assert is_regular(panel) is True

    def test_panel_with_one_irregular_entity_is_irregular(self):
        """One gapped entity makes the global answer False."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]))
        assert is_regular(panel) is False

    def test_per_entity_isolates_the_irregular_entity(self):
        """``per_entity=True`` names the irregular entity and only it."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]))
        assert is_regular(panel, per_entity=True) == {('A',): True, ('B',): False}

    def test_regular_entities_with_different_frequencies_are_globally_irregular(self):
        """Global regularity requires one common frequency."""
        # Entité A mensuelle, entité B trimestrielle, toutes deux sans trou
        quarterly = pd.Series(range(4), index=pd.date_range('2020-01-01', periods=4, freq='QS'))
        panel = _panel(A=_monthly(), B=quarterly)
        assert is_regular(panel) is False

    def test_regular_entities_with_different_frequencies_are_regular_per_entity(self):
        """The same panel is fully regular entity by entity."""
        quarterly = pd.Series(range(4), index=pd.date_range('2020-01-01', periods=4, freq='QS'))
        panel = _panel(A=_monthly(), B=quarterly)
        assert is_regular(panel, per_entity=True) == {('A',): True, ('B',): True}

    def test_start_and_end_positions_are_different_frequencies(self):
        """A month-start entity and a month-end entity do not share a frequency."""
        end = pd.Series(range(5), index=pd.date_range('2020-01-31', periods=5, freq='ME'))
        panel = _panel(A=_monthly(5), B=end)
        assert is_regular(panel) is False

    def test_three_level_panel_keys_keep_every_entity_level(self):
        """With a 3-level index the entity key has two components."""
        series = _monthly()
        panel = pd.concat(
            {('FR', 'x'): series, ('FR', 'y'): series.drop(series.index[2])},
            names=['country', 'sector', 'date'],
        )
        assert is_regular(panel, per_entity=True) == {('FR', 'x'): True, ('FR', 'y'): False}

    def test_unsorted_panel_is_regular_per_entity(self):
        """A shuffled gap-free panel is regular entity by entity."""
        panel = _panel(A=_monthly(), B=_monthly()).sample(frac=1, random_state=0)
        assert is_regular(panel, per_entity=True) == {('A',): True, ('B',): True}

    def test_unsorted_panel_is_globally_regular(self):
        """The global answer ignores row order as well."""
        panel = _panel(A=_monthly(), B=_monthly()).sample(frac=1, random_state=0)
        assert is_regular(panel) is True

    def test_panel_cols_and_time_col_build_the_index(self):
        """``panel_cols`` + ``time_col`` turn a long frame into a panel."""
        series = _monthly()
        dates = list(series.index) + list(series.drop(series.index[2]).index)
        frame = pd.DataFrame({
            'country': ['A'] * 6 + ['B'] * 5,
            'date': dates,
            'x': 1.0,
        })
        result = is_regular(frame, time_col='date', panel_cols=['country'], per_entity=True)
        assert result == {('A',): True, ('B',): False}

    def test_heterogeneous_coverage_panel_is_irregular_for_every_entity(
        self, heterogeneous_coverage_panel
    ):
        """Each entity carries annual anchors before its monthly grid: all irregular."""
        result = is_regular(heterogeneous_coverage_panel, per_entity=True)
        assert result == {('Allemagne',): False, ('France',): False, ('Italie',): False}


# ------------------------------------------------------------------ #
#  regularize : séries                                                #
# ------------------------------------------------------------------ #

class TestRegularizeTimeSeries:
    """Gap filling of a single time series."""

    @pytest.mark.parametrize('freq', _FREQUENCIES)
    def test_gap_is_filled_on_the_regular_grid(self, freq):
        """The result is the full grid between the original bounds."""
        # Valeur d'or : la grille complète d'origine, 8 dates
        grid = pd.date_range('2020-01-01', periods=8, freq=freq)
        series = pd.Series(np.arange(8, dtype=float), index=grid)
        result = regularize(series.drop(grid[3]))
        assert result.index.equals(grid)

    @pytest.mark.parametrize('freq', _FREQUENCIES)
    def test_only_the_added_date_is_nan(self, freq):
        """Original values are preserved, the single added date is NaN."""
        grid = pd.date_range('2020-01-01', periods=8, freq=freq)
        series = pd.Series(np.arange(8, dtype=float), index=grid)
        result = regularize(series.drop(grid[3]))
        expected = series.copy()
        expected.iloc[3] = np.nan
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    @pytest.mark.parametrize('freq', _FREQUENCIES)
    def test_result_is_regular(self, freq):
        """The regularized series is regular for ``is_regular``."""
        grid = pd.date_range('2020-01-01', periods=8, freq=freq)
        series = pd.Series(np.arange(8, dtype=float), index=grid)
        assert is_regular(regularize(series.drop(grid[[2, 5]]))) is True

    def test_several_gaps_and_a_wide_gap(self):
        """A three-month hole yields three NaN rows."""
        series = _monthly(8)
        result = regularize(series.drop(series.index[[2, 3, 4]]))
        assert result.isna().sum() == 3

    def test_regular_series_is_unchanged(self):
        """Nothing to fill: the same values on the same index."""
        series = _monthly()
        pd.testing.assert_series_equal(regularize(series), series, check_freq=False)

    def test_idempotent(self):
        """Regularizing twice equals regularizing once."""
        series = _monthly(8)
        once = regularize(series.drop(series.index[[2, 5]]))
        pd.testing.assert_series_equal(regularize(once), once, check_freq=False)

    def test_unsorted_input_is_sorted_and_filled(self):
        """A shuffled series gives the sorted regular grid."""
        series = _monthly(8)
        gapped = series.drop(series.index[2]).iloc[[4, 0, 3, 1, 5, 2, 6]]
        result = regularize(gapped)
        assert result.index.is_monotonic_increasing and len(result) == 8

    def test_unsorted_complete_series_equals_sorted_series(self):
        """Sorting is part of the contract: a shuffled gap-free series is restored."""
        series = _monthly()
        pd.testing.assert_series_equal(
            regularize(series.iloc[[3, 0, 5, 1, 4, 2]]), series, check_freq=False
        )

    def test_nan_values_are_kept_and_not_confused_with_gaps(self):
        """An original NaN stays NaN; the added date adds one more."""
        series = _monthly()
        series.iloc[4] = np.nan
        result = regularize(series.drop(series.index[1]))
        assert result.isna().sum() == 2

    def test_input_is_not_mutated(self):
        """The caller's object is left untouched."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        snapshot = series.copy()
        regularize(series)
        pd.testing.assert_series_equal(series, snapshot)

    def test_dataframe_columns_are_all_reindexed(self):
        """Every column of a DataFrame gets NaN on the added date."""
        series = _monthly()
        frame = pd.DataFrame({'a b': series, 'é': series * 2}).drop(series.index[2])
        result = regularize(frame)
        assert result.loc['2020-03-01'].isna().all()

    def test_index_name_is_preserved(self):
        """A non-standard index name survives the reindexing."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        series.index.name = 'période'
        assert regularize(series).index.name == 'période'

    def test_duplicated_timestamps_raise(self):
        """Duplicated timestamps cannot be regularized."""
        dates = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-02-01'])
        with pytest.raises(ValueError, match="Duplicated timestamps found in the index"):
            regularize(pd.Series([1.0, 2.0, 3.0], index=dates))

    def test_single_observation_is_returned_unchanged(self):
        """No frequency can be read from one date: the data comes back as is."""
        series = _monthly().iloc[:1]
        pd.testing.assert_series_equal(regularize(series), series)

    def test_empty_series_is_returned_empty(self):
        """An empty series stays empty."""
        assert len(regularize(_monthly().iloc[:0])) == 0

    def test_time_col_round_trip(self):
        """With ``time_col`` the date comes back as a column on a fresh RangeIndex."""
        frame = pd.DataFrame({'date': _monthly().index.delete(2), 'x': [0.0, 1.0, 3.0, 4.0, 5.0]})
        result = regularize(frame, time_col='date')
        expected = pd.DataFrame({
            'date': _monthly().index,
            'x': [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
        })
        pd.testing.assert_frame_equal(result, expected)


class TestRegularizePeriodIndex:
    """A PeriodIndex is regularized on its periods and returned as periods."""

    @pytest.mark.parametrize('freq', ['D', 'M', 'Q', 'Y', 'W'])
    def test_gap_is_filled_and_periods_are_restored(self, freq):
        """Same period frequency in and out, the dropped period comes back as NaN."""
        periods = pd.period_range('2020-01-01', periods=8, freq=freq)
        series = pd.Series(np.arange(8, dtype=float), index=periods)
        result = regularize(series.drop(periods[3]))
        expected = series.copy()
        expected.iloc[3] = np.nan
        pd.testing.assert_series_equal(result, expected)

    def test_index_name_is_preserved(self):
        """The index name survives the round trip through timestamps."""
        periods = pd.period_range('2020-01', periods=6, freq='M', name='mois')
        series = pd.Series(np.arange(6, dtype=float), index=periods).drop(periods[2])
        assert regularize(series).index.name == 'mois'

    def test_panel_last_level_is_restored_as_periods(self):
        """Entities and level names are kept, the time level is a PeriodIndex again."""
        periods = pd.period_range('2020-01', periods=6, freq='M')
        panel = pd.concat({'A': pd.Series(np.arange(6.0), index=periods).drop(periods[2]),
                           'B': pd.Series(np.arange(6.0), index=periods)},
                          names=['entity', 'date'])
        result = regularize(panel)
        assert isinstance(result.index.get_level_values(-1), pd.PeriodIndex)
        assert result.index.names == ['entity', 'date'] and len(result) == 12

    def test_period_time_col_round_trip(self):
        """A column of Periods given as ``time_col`` comes back as a column of Periods."""
        periods = pd.period_range('2020-01', periods=6, freq='M')
        frame = pd.DataFrame({'date': periods.delete(2), 'x': [0.0, 1.0, 3.0, 4.0, 5.0]})
        result = regularize(frame, time_col='date')
        assert result['date'].tolist() == list(periods) and result['x'].isna().sum() == 1

    def test_regularize_does_not_mutate_the_input(self):
        """The caller's PeriodIndex is untouched."""
        periods = pd.period_range('2020-01', periods=6, freq='M')
        series = pd.Series(np.arange(6.0), index=periods).drop(periods[2])
        snapshot = series.copy()
        regularize(series)
        pd.testing.assert_series_equal(series, snapshot)


class TestRegularizeIrregularIndexTimeSeries:
    """Realistic series: monthly grid preceded by isolated annual anchors."""

    def test_result_is_regular_between_the_original_bounds(self, irregular_index_timeseries):
        """The grid runs from the first annual anchor to the last month."""
        result = regularize(irregular_index_timeseries)
        original = irregular_index_timeseries.index
        assert (result.index.min(), result.index.max()) == (original.min(), original.max())
        assert is_regular(result) is True

    def test_no_observation_is_destroyed(self, irregular_index_timeseries):
        """Every original row is found unchanged in the result."""
        result = regularize(irregular_index_timeseries)
        pd.testing.assert_frame_equal(
            result.loc[irregular_index_timeseries.index], irregular_index_timeseries,
            check_freq=False,
        )

    def test_number_of_observations_per_column_is_conserved(self, irregular_index_timeseries):
        """Gap filling adds NaN only: the non-null count of each column is unchanged."""
        result = regularize(irregular_index_timeseries)
        pd.testing.assert_series_equal(result.count(), irregular_index_timeseries.count())


# ------------------------------------------------------------------ #
#  regularize : panels                                                #
# ------------------------------------------------------------------ #

class TestRegularizePanel:
    """Gap filling entity by entity."""

    def test_each_entity_is_filled_on_its_own_grid(self):
        """Only the irregular entity grows; the regular one is untouched."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]))
        result = regularize(panel)
        assert {e: len(result.loc[e]) for e in 'AB'} == {'A': 6, 'B': 6}

    def test_values_are_preserved_and_added_dates_are_nan(self):
        """Golden panel: B gets one NaN at 2020-03-01, all other values identical."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]) + 10)
        result = regularize(panel)
        expected = _panel(A=series, B=(series + 10).where(series.index != '2020-03-01'))
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_each_entity_keeps_its_own_bounds(self):
        """Heterogeneous coverage: entities are not stretched to a common period."""
        # A couvre janvier-juin, B couvre mars-août ; chacune a un trou interne
        a = _monthly(6, '2020-01-01')
        b = _monthly(6, '2020-03-01')
        panel = _panel(A=a.drop(a.index[2]), B=b.drop(b.index[2]))
        result = regularize(panel)
        bounds = {e: (result.loc[e].index.min(), result.loc[e].index.max()) for e in 'AB'}
        assert bounds == {
            'A': (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')),
            'B': (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-08-01')),
        }

    def test_index_names_are_preserved(self):
        """The (entity, date) names survive the reassembly."""
        series = _monthly()
        assert regularize(_panel(A=series.drop(series.index[2]))).index.names == ['entity', 'date']

    def test_three_level_panel(self):
        """A 3-level index is reassembled with its three level names."""
        series = _monthly()
        panel = pd.concat(
            {('FR', 'x'): series.drop(series.index[2]), ('DE', 'x'): series},
            names=['country', 'sector', 'date'],
        )
        result = regularize(panel)
        assert result.index.names == ['country', 'sector', 'date'] and len(result) == 12

    def test_idempotent(self):
        """Regularizing a regularized panel changes nothing."""
        series = _monthly()
        once = regularize(_panel(A=series.drop(series.index[2]), B=series))
        pd.testing.assert_series_equal(regularize(once), once, check_freq=False)

    def test_result_is_regular(self):
        """The regularized panel is regular for ``is_regular``."""
        series = _monthly()
        result = regularize(_panel(A=series.drop(series.index[2]), B=series))
        assert is_regular(result) is True

    def test_unsorted_panel_equals_sorted_panel(self):
        """Row order of the input has no influence on the result."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=series)
        shuffled = panel.sample(frac=1, random_state=0)
        pd.testing.assert_series_equal(regularize(shuffled), regularize(panel), check_freq=False)

    def test_duplicated_timestamps_raise(self):
        """A duplicated (entity, date) pair is rejected."""
        panel = _panel(A=_monthly(), B=_monthly())
        with pytest.raises(ValueError, match="Duplicated timestamps found in the index"):
            regularize(pd.concat([panel, panel.iloc[:1]]))

    def test_panel_cols_and_time_col_are_restored_as_columns(self):
        """Long frame in, long frame out: entity and date columns on a fresh RangeIndex."""
        series = _monthly()
        frame = pd.DataFrame({
            'country': ['A'] * 5,
            'date': series.drop(series.index[2]).index,
            'x': [0.0, 1.0, 3.0, 4.0, 5.0],
        })
        result = regularize(frame, time_col='date', panel_cols=['country'])
        expected = pd.DataFrame({
            'country': ['A'] * 6,
            'date': series.index,
            'x': [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_panel_cols_alone_keep_the_datetime_index(self):
        """Without ``time_col`` the existing DatetimeIndex stays the time level."""
        series = _monthly()
        frame = pd.DataFrame({'country': 'A', 'x': 1.0}, index=series.drop(series.index[2]).index)
        result = regularize(frame, panel_cols=['country'])
        assert result.columns.tolist() == ['country', 'x'] and len(result) == 6

    def test_all_nan_entity_keeps_its_rows(self):
        """An entity made only of NaN is regularized like any other."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=series * np.nan)
        assert len(regularize(panel).loc['B']) == 6


class TestRegularizeMixedPositionsAndFrequencies:
    """Entities of different positions or frequencies, with and without ``per_entity``."""

    @staticmethod
    def _start_and_end_panel() -> pd.Series:
        """Panel of month-start entity A and month-end entity B, both with a gap."""
        start = _monthly(5)
        end = pd.Series(np.arange(5.0), index=pd.date_range('2020-01-31', periods=5, freq='ME'))
        return _panel(A=start.drop(start.index[2]), B=end.drop(end.index[2]))

    def test_inconsistent_positions_raise_in_global_mode(self):
        """One global frequency cannot be both a start and an end grid."""
        with pytest.raises(ValueError, match="Mixed positions detected"):
            regularize(self._start_and_end_panel())

    def test_inconsistent_positions_error_names_the_entities(self):
        """The message lists the position of each entity."""
        with pytest.raises(ValueError, match=r"\('A',\): 'S'.*\('B',\): 'E'"):
            regularize(self._start_and_end_panel())

    def test_per_entity_mode_keeps_each_position(self):
        """With ``per_entity=True`` each entity keeps its own start / end grid."""
        result = regularize(self._start_and_end_panel(), per_entity=True)
        freqs = {e: pd.infer_freq(result.loc[e].index) for e in 'AB'}
        assert freqs == {'A': 'MS', 'B': 'ME'}

    @staticmethod
    def _monthly_and_quarterly_panel() -> pd.Series:
        """Panel of a gapped monthly entity A and a gapped quarterly entity B."""
        monthly = _monthly(5)
        quarterly = pd.Series(np.arange(5.0), index=pd.date_range('2020-01-01', periods=5, freq='QS'))
        return _panel(A=monthly.drop(monthly.index[2]), B=quarterly.drop(quarterly.index[2]))

    def test_per_entity_mode_keeps_each_frequency(self):
        """With ``per_entity=True`` the quarterly entity stays quarterly (5 rows)."""
        result = regularize(self._monthly_and_quarterly_panel(), per_entity=True)
        assert {e: len(result.loc[e]) for e in 'AB'} == {'A': 5, 'B': 5}

    def test_global_mode_rewrites_the_quarterly_entity_on_the_monthly_grid(self):
        """Global mode is an alignment on one frequency: the quarterly entity becomes monthly."""
        # Valeur d'or : B passe de 5 trimestres (2020Q1-2021Q1) à 13 mois
        result = regularize(self._monthly_and_quarterly_panel())
        assert len(result.loc['B']) == 13 and pd.infer_freq(result.loc['B'].index) == 'MS'

    def test_global_mode_does_not_destroy_the_quarterly_observations(self):
        """Even rewritten on the monthly grid, the 4 quarterly observations remain."""
        result = regularize(self._monthly_and_quarterly_panel())
        assert result.loc['B'].count() == 4


class TestRegularizeHeterogeneousCoveragePanel:
    """Realistic panel: coverage and annual anchors specific to each entity."""

    def test_each_entity_keeps_its_original_bounds(self, heterogeneous_coverage_panel):
        """No entity is stretched or truncated."""
        result = regularize(heterogeneous_coverage_panel)
        for entity, block in heterogeneous_coverage_panel.groupby(level=0):
            dates = block.index.get_level_values(-1)
            kept = result.loc[entity].index
            assert (kept.min(), kept.max()) == (dates.min(), dates.max())

    def test_no_observation_is_destroyed(self, heterogeneous_coverage_panel):
        """Every original row is found unchanged in the result."""
        result = regularize(heterogeneous_coverage_panel)
        pd.testing.assert_frame_equal(
            result.loc[heterogeneous_coverage_panel.index], heterogeneous_coverage_panel
        )

    def test_non_null_count_per_entity_and_column_is_conserved(self, heterogeneous_coverage_panel):
        """Only NaN rows are added, per (entity, column) pair."""
        result = regularize(heterogeneous_coverage_panel)
        before = heterogeneous_coverage_panel.groupby(level=0).count()
        after = result.groupby(level=0).count()
        pd.testing.assert_frame_equal(after, before)

    @pytest.mark.parametrize('per_entity', [False, True], ids=['global', 'per_entity'])
    def test_result_is_regular_for_every_entity(self, heterogeneous_coverage_panel, per_entity):
        """After regularization every entity is regular."""
        result = regularize(heterogeneous_coverage_panel, per_entity=per_entity)
        assert all(is_regular(result, per_entity=True).values())

    def test_entity_level_frequency_is_monthly(self, heterogeneous_coverage_panel):
        """The dominant monthly grid is the one rebuilt for each entity."""
        result = regularize(heterogeneous_coverage_panel)
        freqs = {e: pd.infer_freq(result.loc[e].index) for e in result.index.get_level_values(0).unique()}
        assert set(freqs.values()) == {'MS'}


# ------------------------------------------------------------------ #
#  Équivalence fonctions / méthodes                                   #
# ------------------------------------------------------------------ #

class TestFunctionsMatchMethods:
    """Module functions are thin wrappers of ``IndexRegularizer`` methods."""

    @pytest.mark.parametrize('per_entity', [False, True], ids=['global', 'per_entity'])
    def test_is_regular_equivalence(self, per_entity):
        """``is_regular`` equals ``IndexRegularizer().is_regular``."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]))
        assert is_regular(panel, per_entity=per_entity) == IndexRegularizer().is_regular(
            panel, per_entity=per_entity
        )

    @pytest.mark.parametrize('per_entity', [False, True], ids=['global', 'per_entity'])
    def test_regularize_equivalence(self, per_entity):
        """``regularize`` equals ``IndexRegularizer().regularize``."""
        series = _monthly()
        panel = _panel(A=series, B=series.drop(series.index[2]))
        pd.testing.assert_series_equal(
            regularize(panel, per_entity=per_entity),
            IndexRegularizer().regularize(panel, per_entity=per_entity),
        )

    def test_positional_arguments_are_forwarded_in_order(self):
        """Positional ``time_col, panel_cols, per_entity`` reach the method in order."""
        series = _monthly()
        frame = pd.DataFrame({
            'country': ['A'] * 5, 'date': series.drop(series.index[2]).index, 'x': 1.0,
        })
        assert is_regular(frame, 'date', ['country'], True) == {('A',): False}

    def test_method_instances_are_stateless(self):
        """Two instances (or two calls) give the same answer: no fitted state."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        first, second = IndexRegularizer(), IndexRegularizer()
        pd.testing.assert_series_equal(first.regularize(series), second.regularize(series))


# ------------------------------------------------------------------ #
#  Observations hors grille et fréquences indétectables               #
# ------------------------------------------------------------------ #

# Trois dates d'espacements 2 et 48 jours : aucune fréquence ne domine
_UNDETECTABLE_DATES = pd.to_datetime(['2020-01-01', '2020-01-03', '2020-02-20'])


class TestOffGridObservations:
    """Observations off the detected grid: dropped with a warning, or an error."""

    @staticmethod
    def _mid_month_series() -> pd.Series:
        """Monthly series whose April observation is dated mid-month (15th)."""
        dates = list(pd.date_range('2020-01-01', periods=10, freq='MS'))
        dates[3] = pd.Timestamp('2020-04-15')
        return pd.Series(np.arange(10, dtype=float), index=dates)

    def test_mid_month_outlier_keeps_the_month_start_grid(self):
        """The 9 month-start dates give an 'MS' grid; the 15 April observation is dropped."""
        # Valeur d'or : 10 mois, 9 observations sur la grille, 04-01 rempli en NaN
        with pytest.warns(UserWarning, match=r"1 observation\(s\) not on the 'MS' grid dropped: 2020-04-15"):
            result = regularize(self._mid_month_series())
        assert len(result) == 10 and result.count() == 9 and pd.isna(result['2020-04-01'])

    def test_no_observation_on_the_grid_raises(self):
        """Dates scattered inside the months: no position, no overlap with any grid."""
        dates = pd.to_datetime(['2020-01-10', '2020-02-15', '2020-03-20', '2020-04-12', '2020-05-18'])
        with pytest.raises(ValueError):
            regularize(pd.Series(1.0, index=dates))

    def test_some_observations_off_the_grid_are_dropped_with_a_warning(self):
        """Daily grid detected: the single noon observation is dropped and listed."""
        # Valeur d'or : 30 jours sur la grille, 1 observation à midi écartée
        regular = pd.date_range('2020-01-01', periods=30, freq='D')
        series = pd.Series(1.0, index=regular.append(pd.DatetimeIndex(['2020-01-05 12:00'])).sort_values())
        with pytest.warns(UserWarning, match=r"1 observation\(s\) not on the 'D' grid dropped: 2020-01-05 12:00"):
            result = regularize(series)
        assert result.count() == 30 and len(result) == 30

    def test_off_grid_warning_lists_at_most_five_dates(self):
        """Seven off-grid dates: five listed, the rest counted."""
        regular = pd.date_range('2020-01-01', periods=30, freq='D')
        off = pd.DatetimeIndex([f'2020-01-{d:02d} 12:00' for d in range(2, 9)])
        series = pd.Series(1.0, index=regular.append(off).sort_values())
        with pytest.warns(UserWarning, match=r"7 observation\(s\).*\(\+2 more\)"):
            regularize(series)

    def test_panel_warning_names_the_entity(self):
        """In a panel the warning identifies the entity concerned."""
        regular = pd.date_range('2020-01-01', periods=30, freq='D')
        off = pd.DatetimeIndex(['2020-01-05 12:00'])
        noisy = pd.Series(1.0, index=regular.append(off).sort_values())
        panel = _panel(A=pd.Series(1.0, index=regular), B=noisy)
        with pytest.warns(UserWarning, match=r"Entity \('B',\): 1 observation"):
            regularize(panel)

    def test_clean_input_emits_no_warning(self):
        """A gap-only series is regularized silently."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            regularize(series)


class TestUndetectableFrequency:
    """No constant frequency can be built: an error, never an irregular output."""

    @staticmethod
    def _undetectable() -> pd.Series:
        """Three observations on dates with no dominant spacing."""
        return pd.Series([1.0, 2.0, 3.0], index=_UNDETECTABLE_DATES)

    def test_series_raises(self):
        """A series with no detectable frequency cannot be regularized."""
        with pytest.raises(ValueError, match="no constant frequency can be detected"):
            regularize(self._undetectable())

    def test_panel_without_any_detectable_entity_raises(self):
        """Global mode needs at least one detectable entity."""
        panel = _panel(A=self._undetectable(), B=self._undetectable())
        with pytest.raises(ValueError, match=r"Entity \('A',\): no constant frequency"):
            regularize(panel)

    def test_per_entity_mode_raises_naming_the_undetectable_entity(self):
        """In ``per_entity`` mode the offending entity is named."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=self._undetectable())
        with pytest.raises(ValueError, match=r"Entity \('B',\): no constant frequency"):
            regularize(panel, per_entity=True)

    def test_global_mode_raises_as_well(self):
        """The undetectable entity is not silently squeezed onto its neighbors' grid."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=self._undetectable())
        with pytest.raises(ValueError, match=r"Entity \('B',\): no constant frequency"):
            regularize(panel)

    def test_single_observation_entity_is_kept_as_is(self):
        """One date is not a grid problem: the entity is left untouched in both modes."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=series.iloc[:1])
        result = regularize(panel)
        assert len(result.loc['A']) == 6 and len(result.loc['B']) == 1

    def test_panel_of_single_observation_entities_is_returned_unchanged(self):
        """No entity has a grid: nothing to regularize, nothing to raise."""
        series = _monthly().iloc[:1]
        panel = _panel(A=series, B=series)
        pd.testing.assert_series_equal(regularize(panel), panel)

    def test_single_observation_entity_is_kept_in_per_entity_mode(self):
        """Same in ``per_entity`` mode."""
        series = _monthly()
        panel = _panel(A=series.drop(series.index[2]), B=series.iloc[:1])
        assert len(regularize(panel, per_entity=True).loc['B']) == 1


class TestEmptyAndDegenerateInputs:
    """Empty panels and empty ``panel_cols``."""

    def test_empty_panel_is_returned_empty(self):
        """A panel without any row stays empty."""
        panel = _panel(A=_monthly())
        assert len(regularize(panel.iloc[:0])) == 0

    def test_empty_panel_cols_list_is_ignored(self):
        """``panel_cols=[]`` does not build any index level."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        assert len(regularize(series.to_frame('x'), panel_cols=[])) == 6

    def test_integer_index_raises(self):
        """Integer labels are rejected with an explicit error."""
        with pytest.raises(TypeError, match="must be a DatetimeIndex or a PeriodIndex"):
            regularize(pd.Series(range(4), index=pd.RangeIndex(4)))

    def test_regularize_emits_no_deprecation_warning(self):
        """The grid is built with non-deprecated pandas aliases."""
        series = _monthly().drop(pd.Timestamp('2020-03-01'))
        series.index = series.index + pd.offsets.MonthEnd(0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            regularize(series)

    @pytest.mark.internal
    def test_validate_consistent_positions_ignores_frequencies_without_position(self):
        """A frequency without position (daily) does not take part in the check."""
        assert IndexRegularizer._validate_consistent_positions({('A',): 'MS', ('B',): 'D'}) is None
