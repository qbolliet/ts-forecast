"""Tests for panel detection (``utils/frequency/detector.py``).

Covers ``FrequencyDetector.detect_frequency`` (simple series, or one frequency per
entity of a ``MultiIndex`` series, keys always tuples since commit ``67529a5``) and
``FrequencyDetector.detect_dataset_frequency`` (one frequency per column of a time
series, or per ``(entity..., column)`` pair of a panel, keys flattened; undetectable
pairs mapped to ``None`` since commit ``906da2e``): 2 and 3-level indexes, panel
columns, ``time_col``, unsorted dates, unnamed levels (ANO-UTILS-046), invalid
``return_format`` (ANO-UTILS-047), undetectable time series columns (ANO-UTILS-042).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.unit.utils.frequency.detector.helpers import make_panel_series, make_series
from tsforecast.utils.frequency.detector import FrequencyDetector


# =============================================================================
# FrequencyDetector.detect_frequency : série simple ou à MultiIndex
# =============================================================================

class TestDetectorDetectFrequencyPanelSeries:
    """A panel series gives one frequency per entity, keyed by a tuple (commit ``67529a5``)."""

    def test_simple_series_returns_string(self):
        """Without ``MultiIndex``, the frequency itself is returned."""
        series = make_series(pd.date_range('2023-01-01', periods=5, freq='D'))
        assert FrequencyDetector().detect_frequency(series) == 'D'

    def test_two_level_index(self):
        """Single entity level: keys are one-element tuples ``('A',)``."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D'),
                                'B': pd.date_range('2023-01-01', periods=3, freq='D')})
        assert FrequencyDetector().detect_frequency(series) == {('A',): 'D', ('B',): 'D'}

    def test_two_level_index_emits_no_future_warning(self):
        """A single entity level is grouped without pandas' ``FutureWarning``."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D'),
                                'B': pd.date_range('2023-01-01', periods=3, freq='D')})
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            detected = FrequencyDetector().detect_frequency(series)
        assert detected == {('A',): 'D', ('B',): 'D'}

    def test_three_level_index(self):
        """Two entity levels: keys are ``(country, region)`` tuples."""
        dates = pd.date_range('2023-01-01', periods=2, freq='D').tolist()
        index = pd.MultiIndex.from_arrays(
            [['X'] * 4 + ['Y'] * 4, ['A', 'A', 'B', 'B'] * 2, dates * 4],
            names=['country', 'region', 'date'],
        )
        expected = {('X', 'A'): 'D', ('X', 'B'): 'D', ('Y', 'A'): 'D', ('Y', 'B'): 'D'}
        assert FrequencyDetector().detect_frequency(make_series(index)) == expected

    def test_unsorted_dates_within_entities(self):
        """Dates unsorted inside each entity are still daily."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        series = make_panel_series({'A': [dates[2], dates[0], dates[1]],
                                'B': [dates[1], dates[2], dates[0]]})
        assert FrequencyDetector().detect_frequency(series) == {('A',): 'D', ('B',): 'D'}

    def test_heterogeneous_frequency_per_entity(self):
        """Each entity keeps its own frequency."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=4, freq='MS'),
                                'B': pd.date_range('2023-01-01', periods=4, freq='QS')})
        detected = FrequencyDetector().detect_frequency(series, 'with_position')
        assert detected == {('A',): 'MS', ('B',): 'QS'}

    def test_entity_with_too_few_observations_is_kept_as_none(self):
        """An entity with one date stays in the map, mapped to ``None`` (commit ``906da2e``)."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D'),
                                'B': ['2023-01-01']})
        detected = FrequencyDetector(min_observations=2).detect_frequency(series)
        assert detected == {('A',): 'D', ('B',): None}

    def test_all_entities_with_too_few_observations(self):
        """Every entity is kept, mapped to ``None``: nothing is silently dropped."""
        series = make_panel_series({'A': ['2023-01-01'], 'B': ['2023-01-02']})
        assert FrequencyDetector(min_observations=2).detect_frequency(series) == {('A',): None,
                                                                                  ('B',): None}

    def test_unnamed_levels(self):
        """Unnamed index levels are grouped by position."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D').tolist()
        index = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates * 2])
        assert FrequencyDetector().detect_frequency(make_series(index)) == {('A',): 'D', ('B',): 'D'}

    def test_empty_panel_series_is_none(self):
        """A panel series without rows has no entity: ``None``."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D')}).iloc[0:0]
        assert FrequencyDetector().detect_frequency(series) is None

    def test_single_level_multiindex_raises(self):
        """A one-level ``MultiIndex`` has no date level: ``ValueError``."""
        series = pd.Series([1, 2, 3], index=pd.MultiIndex.from_arrays([['A', 'B', 'C']]))
        with pytest.raises(ValueError, match='at least 2 levels'):
            FrequencyDetector().detect_frequency(series)

    def test_invalid_return_format_raises(self):
        """An unknown ``return_format`` is rejected on a panel as on a simple series (ANO-UTILS-047)."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D')})
        with pytest.raises(ValueError, match='Invalid return_format'):
            FrequencyDetector().detect_frequency(series, 'bogus')

    def test_non_date_level_raises(self):
        """A date level that is not a time index raises instead of mapping entities to ``None``."""
        # Dernier niveau entier (années) : refusé comme une série simple (ANO-UTILS-044)
        index = pd.MultiIndex.from_arrays([['A'] * 3, [2020, 2021, 2022]], names=['entity', 'year'])
        with pytest.raises(ValueError, match='cannot be converted to datetime'):
            FrequencyDetector().detect_frequency(make_series(index))


# =============================================================================
# FrequencyDetector.detect_dataset_frequency
# =============================================================================

class TestDetectDatasetFrequencyTimeSeries:
    """A plain ``DataFrame`` gives one frequency per column, keyed by column name."""

    def test_columns_on_the_index(self):
        """Every column of a daily frame is daily."""
        df = pd.DataFrame({'value1': [1, 2, 3, 4, 5], 'value2': [10, 20, 30, 40, 50]},
                          index=pd.date_range('2023-01-01', periods=5, freq='D'))
        assert FrequencyDetector().detect_dataset_frequency(df) == {'value1': 'D', 'value2': 'D'}

    def test_columns_with_their_own_frequency(self):
        """Columns observed at different frequencies on a shared grid keep their own frequency."""
        # Grille mensuelle commune ; 'q' observée un mois sur trois (trimestres)
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'m': np.arange(12.0),
                           'q': [1.0 if month in (1, 4, 7, 10) else np.nan for month in index.month]},
                          index=index)
        detected = FrequencyDetector().detect_dataset_frequency(df, return_format='with_position')
        assert detected == {'m': 'MS', 'q': 'QS'}

    @pytest.mark.parametrize(
        'dates',
        [pytest.param(pd.date_range('2024-01-01', periods=4, freq='QS'), id='sorted'),
         pytest.param(pd.date_range('2024-01-01', periods=4, freq='QS')[::-1], id='reversed')],
    )
    def test_time_column(self, dates):
        """``time_col`` supplies the dates and is not reported as a variable."""
        df = pd.DataFrame({'date': dates, 'v': range(4)})
        detected = FrequencyDetector().detect_dataset_frequency(df, time_col='date',
                                                                return_format='with_position')
        assert detected == {'v': 'QS'}

    def test_missing_time_column_raises(self):
        """A ``time_col`` absent from the columns raises instead of being ignored."""
        # Le RangeIndex résiduel ne doit pas être pris à tort pour des dates
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]},
                          index=pd.date_range('2023-01-01', periods=5, freq='D')).reset_index(names='date')
        with pytest.raises(ValueError, match="time_col 'not_a_column' not found"):
            FrequencyDetector().detect_dataset_frequency(df, time_col='not_a_column')

    def test_unsorted_index(self):
        """An unsorted index gives the frequency of the sorted one."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=[dates[i] for i in [2, 0, 4, 1, 3]])
        assert FrequencyDetector().detect_dataset_frequency(df) == {'value': 'D'}

    def test_invalid_return_format_raises(self):
        """An unknown ``return_format`` is rejected."""
        df = pd.DataFrame({'v': range(5)}, index=pd.date_range('2024-01-01', periods=5))
        with pytest.raises(ValueError, match='Invalid return_format'):
            FrequencyDetector().detect_dataset_frequency(df, return_format='bogus')

    def test_column_with_too_few_observations_is_none(self):
        """A column observed once is mapped to ``None``, as a panel pair is (ANO-UTILS-042)."""
        df = pd.DataFrame({'dense': np.arange(5.0), 'sparse': [np.nan] * 4 + [1.0]},
                          index=pd.date_range('2024-01-01', periods=5, freq='D'))
        assert FrequencyDetector().detect_dataset_frequency(df) == {'dense': 'D', 'sparse': None}

    def test_empty_frame_maps_every_column_to_none(self):
        """A frame without rows keeps its columns, all undetectable."""
        df = pd.DataFrame({'a': [], 'b': []}, index=pd.DatetimeIndex([]), dtype=float)
        assert FrequencyDetector().detect_dataset_frequency(df) == {'a': None, 'b': None}

    def test_column_with_unrecognized_spacing_is_none(self):
        """A column observed on irregular dates is mapped to ``None`` rather than omitted."""
        # Observations aux jours 0, 45 et 95 : écarts 45 et 50 jours, aucune fréquence
        index = pd.date_range('2024-01-01', periods=100, freq='D')
        irregular = pd.Series(np.nan, index=index)
        irregular.iloc[[0, 45, 95]] = 1.0
        df = pd.DataFrame({'dense': np.arange(100.0), 'irregular': irregular})
        assert FrequencyDetector().detect_dataset_frequency(df) == {'dense': 'D', 'irregular': None}


class TestDetectDatasetFrequencyPanel:
    """A panel gives one frequency per pair, keyed by a FLAT ``(entity..., column)`` tuple."""

    @pytest.fixture
    def two_entity_panel(self) -> pd.DataFrame:
        """Two daily entities on a named ``(panel_id, date)`` index."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D').tolist()
        index = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates * 2],
                                          names=['panel_id', 'date'])
        return pd.DataFrame({'x': range(6), 'y': np.arange(6.0)}, index=index)

    def test_multiindex_auto_detection(self, two_entity_panel):
        """A two-level index is read as a panel without ``panel_cols``."""
        expected = {('A', 'x'): 'D', ('A', 'y'): 'D', ('B', 'x'): 'D', ('B', 'y'): 'D'}
        assert FrequencyDetector().detect_dataset_frequency(two_entity_panel) == expected

    def test_single_level_panel_emits_no_future_warning(self, two_entity_panel):
        """A single entity level is grouped without pandas' ``FutureWarning``."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            detected = FrequencyDetector().detect_dataset_frequency(two_entity_panel[['x']])
        assert detected == {('A', 'x'): 'D', ('B', 'x'): 'D'}

    def test_three_level_index_keys_are_flat(self):
        """Two entity levels are spliced into the key: ``('X', 'A', 'value')``."""
        index = pd.MultiIndex.from_arrays(
            [['X', 'X', 'Y', 'Y'], ['A', 'A', 'B', 'B'],
             pd.date_range('2023-01-01', periods=2, freq='D').tolist() * 2],
            names=['country', 'region', 'date'],
        )
        df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index)
        expected = {('X', 'A', 'value'): 'D', ('Y', 'B', 'value'): 'D'}
        assert FrequencyDetector().detect_dataset_frequency(df) == expected

    def test_panel_columns(self):
        """Panel and time given as columns yield the same flat keys as an index."""
        df = pd.DataFrame({
            'panel_id': ['A'] * 3 + ['B'] * 3,
            'date': pd.date_range('2023-01-01', periods=3, freq='D').tolist() * 2,
            'value': [1, 2, 3, 4, 5, 6],
        })
        detected = FrequencyDetector().detect_dataset_frequency(df, time_col='date',
                                                                panel_cols=['panel_id'])
        assert detected == {('A', 'value'): 'D', ('B', 'value'): 'D'}

    def test_explicit_panel_levels_equal_auto_detection(self, two_entity_panel):
        """Naming the entity level explicitly gives the auto-detected map."""
        detector = FrequencyDetector()
        assert (detector.detect_dataset_frequency(two_entity_panel, panel_cols=['panel_id'])
                == detector.detect_dataset_frequency(two_entity_panel))

    def test_empty_panel_columns_fall_back_on_series_detection(self, two_entity_panel):
        """``panel_cols=[]`` reads each column as a panel series: same flat keys."""
        detector = FrequencyDetector()
        assert (detector.detect_dataset_frequency(two_entity_panel, panel_cols=[])
                == detector.detect_dataset_frequency(two_entity_panel))

    def test_panel_columns_next_to_a_foreign_multiindex(self):
        """Panel columns win over an index level that is not an entity."""
        # Index à deux niveaux (clé constante, date) ; l'entité est la colonne 'ent'
        index = pd.MultiIndex.from_arrays(
            [['k'] * 6, pd.date_range('2024-01-01', periods=6, freq='D')], names=['key', 'date'])
        df = pd.DataFrame({'ent': ['A'] * 3 + ['B'] * 3, 'v': range(6)}, index=index)
        detected = FrequencyDetector().detect_dataset_frequency(df, panel_cols=['ent'])
        assert detected == {('A', 'v'): 'D', ('B', 'v'): 'D'}

    def test_pair_with_too_few_observations_is_none(self, two_entity_panel):
        """A pair observed once stays in the map, mapped to ``None`` (commit ``906da2e``)."""
        two_entity_panel['y'] = [1.0, 2.0, 3.0, np.nan, np.nan, 6.0]
        detected = FrequencyDetector().detect_dataset_frequency(two_entity_panel)
        assert detected == {('A', 'x'): 'D', ('A', 'y'): 'D', ('B', 'x'): 'D', ('B', 'y'): None}

    def test_empty_panel(self, two_entity_panel):
        """A panel without rows has no pair: empty map."""
        assert FrequencyDetector().detect_dataset_frequency(two_entity_panel.iloc[0:0]) == {}

    def test_single_level_multiindex_raises(self):
        """A one-level ``MultiIndex`` frame raises (no date level)."""
        index = pd.MultiIndex.from_arrays([pd.date_range('2024-01-01', periods=3)])
        with pytest.raises(ValueError, match='at least 2 levels'):
            FrequencyDetector().detect_dataset_frequency(pd.DataFrame({'v': [1, 2, 3]}, index=index))

    @pytest.mark.parametrize(
        'names, expected',
        [
            pytest.param([None, None], {('A', 'v'): 'D', ('B', 'v'): 'D'}, id='two-unnamed-levels'),
            # Trois niveaux, le premier seul sans nom : clés aplaties (région, entité, colonne)
            pytest.param([None, 'entity', 'date'], {('R', 'A', 'v'): 'D', ('R', 'B', 'v'): 'D'},
                         id='partly-unnamed-three-levels'),
        ],
    )
    def test_unnamed_levels(self, names, expected):
        """Unnamed index levels are read by position, as for a panel series (ANO-UTILS-046)."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D').tolist()
        arrays = [['A'] * 3 + ['B'] * 3, dates * 2]
        if len(names) == 3:
            arrays.insert(0, ['R'] * 6)
        index = pd.MultiIndex.from_arrays(arrays, names=names)
        df = pd.DataFrame({'v': range(6)}, index=index)
        assert FrequencyDetector().detect_dataset_frequency(df) == expected

    def test_period_dates(self):
        """A monthly ``Period`` date level is monthly (ANO-UTILS-045, ANO-UTILS-029 decision)."""
        index = pd.MultiIndex.from_arrays(
            [['A'] * 4, pd.period_range('2024-01', periods=4, freq='M')], names=['entity', 'date'])
        df = pd.DataFrame({'v': range(4)}, index=index)
        assert FrequencyDetector().detect_dataset_frequency(df, return_format='with_position') == {('A', 'v'): 'MS'}

    def test_invalid_return_format_raises(self, two_entity_panel):
        """An unknown ``return_format`` is rejected on a panel as on a time series (ANO-UTILS-047)."""
        with pytest.raises(ValueError, match='Invalid return_format'):
            FrequencyDetector().detect_dataset_frequency(two_entity_panel, return_format='bogus')
