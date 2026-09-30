"""Tests for frequency consistency (``utils/frequency/detector.py`` and ``utils.py``).

Covers ``FrequencyDetector.validate_frequency_consistency``, the ``check_consistency`` /
``consistency_mode`` / ``strict`` options of the ``detect_frequency`` and
``detect_dataset_frequency`` wrappers (``utils/frequency/utils.py``), the private
``_get_highest_frequency``, and the pinned treatment of undetectable (``None``)
pairs (ANO-UTILS-049, to be arbitrated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.utils.frequency.detector.helpers import make_panel_series, make_series
from tsforecast.utils.frequency.detector import FrequencyDetector
from tsforecast.utils.frequency.utils import (
    _get_highest_frequency,
    detect_dataset_frequency,
    detect_frequency,
)


# =============================================================================
# Cohérence des fréquences
# =============================================================================

class TestValidateFrequencyConsistency:
    """``validate_frequency_consistency`` reduces a frequency map to one frequency."""

    @pytest.mark.parametrize(
        'frequency_map, strict, expected',
        [
            pytest.param({}, True, (False, None), id='empty'),
            pytest.param({'a': 'D', 'b': 'D'}, True, (True, 'D'), id='identical'),
            pytest.param({'a': 'D', 'b': 'M'}, True, (False, None), id='strict-mismatch'),
            # Valeur d'or : 'D' deux fois, 'M' une fois -> 'D' modal
            pytest.param({'a': 'D', 'b': 'M', 'c': 'D'}, False, (True, 'D'), id='modal'),
        ],
    )
    def test_reduction(self, frequency_map, strict, expected):
        """Golden (consistent, frequency) pair of each map."""
        detector = FrequencyDetector()
        assert detector.validate_frequency_consistency(frequency_map, strict=strict) == expected


class TestDetectFrequencyFunction:
    """``detect_frequency`` wraps the detector and optionally reduces a panel map."""

    @pytest.fixture
    def daily_and_weekly_panel(self) -> pd.Series:
        """Two daily entities and one weekly entity."""
        return make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D'),
                              'B': pd.date_range('2023-01-01', periods=3, freq='D'),
                              'C': pd.date_range('2023-01-01', periods=3, freq='W')})

    def test_simple_series(self):
        """A simple series gives its frequency."""
        assert detect_frequency(make_series(pd.date_range('2023-01-01', periods=5, freq='D'))) == 'D'

    def test_panel_series_without_consistency(self, daily_and_weekly_panel):
        """Without ``check_consistency``, the per-entity map is returned with tuple keys."""
        assert detect_frequency(daily_and_weekly_panel) == {('A',): 'D', ('B',): 'D', ('C',): 'W'}

    @pytest.mark.parametrize(
        'options, expected',
        [
            pytest.param({'strict': True}, None, id='strict'),
            # Valeur d'or : deux entités journalières contre une hebdomadaire
            pytest.param({'strict': False, 'consistency_mode': 'modal'}, 'D', id='modal'),
            pytest.param({'consistency_mode': 'highest'}, 'D', id='highest'),
        ],
    )
    def test_panel_series_with_consistency(self, daily_and_weekly_panel, options, expected):
        """Golden reduction of a daily / weekly panel in each mode."""
        assert detect_frequency(daily_and_weekly_panel, check_consistency=True, **options) == expected

    def test_consistentmake_panel_series(self):
        """Identical entity frequencies reduce to that frequency, even in strict mode."""
        series = make_panel_series({'A': pd.date_range('2023-01-01', periods=3, freq='D'),
                                'B': pd.date_range('2023-01-01', periods=3, freq='D')})
        assert detect_frequency(series, check_consistency=True) == 'D'

    def test_consistency_is_ignored_on_simple_series(self):
        """``check_consistency`` has nothing to reduce on a simple series."""
        series = make_series(pd.date_range('2023-01-01', periods=5, freq='MS'))
        assert detect_frequency(series, check_consistency=True, return_format='full') == 'MS'

    def test_dataframe_is_delegated(self):
        """A ``DataFrame`` is delegated to ``detect_dataset_frequency``."""
        df = pd.DataFrame({'a': range(5)}, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        assert detect_frequency(df, check_consistency=True) == 'D'

    @pytest.mark.parametrize(
        'options',
        [pytest.param({'time_col': 'date'}, id='time_col'),
         pytest.param({'panel_cols': ['id']}, id='panel_cols')],
    )
    def test_dataframe_options_on_series_raise(self, options):
        """``time_col`` and ``panel_cols`` are reserved to ``DataFrame`` inputs."""
        series = make_series(pd.date_range('2023-01-01', periods=5, freq='D'))
        with pytest.raises(ValueError, match='only valid for DataFrame'):
            detect_frequency(series, **options)

    def test_invalid_consistency_mode_raises(self, daily_and_weekly_panel):
        """An unknown ``consistency_mode`` is rejected."""
        with pytest.raises(ValueError, match='Invalid consistency_mode'):
            detect_frequency(daily_and_weekly_panel, check_consistency=True, consistency_mode='x')

    def test_non_pandas_input_raises(self):
        """Only a ``Series`` or a ``DataFrame`` is accepted."""
        with pytest.raises(ValueError, match='must be a pandas Series or DataFrame'):
            detect_frequency([1, 2, 3])

    def test_single_observation_raises(self):
        """A simple series with one date raises (no map to hold a ``None``)."""
        with pytest.raises(ValueError, match='only 1 non-null observations'):
            detect_frequency(make_series(pd.DatetimeIndex(['2024-01-01'])))


class TestDetectDatasetFrequencyFunction:
    """``detect_dataset_frequency`` optionally reduces the map of a frame."""

    @pytest.fixture
    def mixed_columns(self) -> pd.DataFrame:
        """Two daily columns and one weekly column on a daily grid."""
        # Colonne hebdomadaire : une valeur tous les 7 jours sur la grille journalière
        index = pd.date_range('2023-01-01', periods=28, freq='D')
        weekly = pd.Series(np.nan, index=index)
        weekly.iloc[::7] = 1.0
        return pd.DataFrame({'daily1': np.arange(28.0), 'daily2': np.arange(28.0), 'weekly': weekly})

    @pytest.mark.parametrize(
        'options, expected',
        [
            pytest.param({}, {'daily1': 'D', 'daily2': 'D', 'weekly': 'W'}, id='no-reduction'),
            pytest.param({'check_consistency': True}, None, id='strict'),
            pytest.param({'check_consistency': True, 'strict': False}, 'D', id='modal'),
            pytest.param({'check_consistency': True, 'consistency_mode': 'highest'}, 'D', id='highest'),
        ],
    )
    def test_reduction(self, mixed_columns, options, expected):
        """Golden result of each consistency option."""
        assert detect_dataset_frequency(mixed_columns, **options) == expected

    def test_consistent_columns(self):
        """Identical column frequencies reduce to that frequency."""
        df = pd.DataFrame({'value1': range(5), 'value2': range(5)},
                          index=pd.date_range('2023-01-01', periods=5, freq='D'))
        assert detect_dataset_frequency(df, check_consistency=True) == 'D'

    def test_invalid_consistency_mode_raises(self, mixed_columns):
        """An unknown ``consistency_mode`` is rejected."""
        with pytest.raises(ValueError, match='Invalid consistency_mode'):
            detect_dataset_frequency(mixed_columns, check_consistency=True, consistency_mode='x')


class TestConsistencyWithUndetectablePairs:
    """Undetectable (``None``) pairs are ignored by consistency checks (ANO-UTILS-049)."""

    @pytest.fixture
    def one_undetectable_entity(self) -> pd.Series:
        """Entity A daily, entity B observed once (``None``)."""
        return make_panel_series({'A': pd.date_range('2024-01-01', periods=3, freq='D'),
                                  'B': ['2024-01-01']})

    @pytest.mark.parametrize(
        'options',
        [pytest.param({'strict': True}, id='strict'),
         pytest.param({'strict': False}, id='modal'),
         pytest.param({'consistency_mode': 'highest'}, id='highest')],
    )
    def test_undetectable_entity_is_ignored(self, one_undetectable_entity, options):
        """An entity observed once neither breaks nor changes the common frequency."""
        assert detect_frequency(one_undetectable_entity, check_consistency=True, **options) == 'D'

    def test_undetectable_majority_does_not_win(self):
        """``None`` is never the modal frequency, even when most entities are undetectable."""
        series = make_panel_series({'A': ['2024-01-01'], 'B': ['2024-01-01'],
                                    'C': pd.date_range('2024-01-01', periods=3, freq='D')})
        assert detect_frequency(series, check_consistency=True, strict=False) == 'D'

    def test_strict_mode_still_detects_a_real_mismatch(self):
        """Ignoring ``None`` keeps a genuine disagreement inconsistent."""
        series = make_panel_series({'A': pd.date_range('2024-01-01', periods=3, freq='D'),
                                    'B': pd.date_range('2024-01-01', periods=3, freq='MS'),
                                    'C': ['2024-01-01']})
        assert detect_frequency(series, check_consistency=True) is None

    @pytest.mark.parametrize(
        'frequency_map, strict, expected',
        [
            pytest.param({'a': None, 'b': None}, True, (False, None), id='only-none'),
            pytest.param({'a': 'D', 'b': None}, True, (True, 'D'), id='strict-with-none'),
            pytest.param({'a': None, 'b': None, 'c': 'M'}, False, (True, 'M'), id='modal-with-none'),
        ],
    )
    def test_validate_frequency_consistency(self, frequency_map, strict, expected):
        """Golden (consistent, frequency) pair of maps holding ``None``."""
        detector = FrequencyDetector()
        assert detector.validate_frequency_consistency(frequency_map, strict=strict) == expected


@pytest.mark.internal
class TestGetHighestFrequency:
    """``_get_highest_frequency`` picks the finest frequency of a map."""

    @pytest.mark.parametrize(
        'frequency_map, expected',
        [
            pytest.param({}, None, id='empty'),
            pytest.param({'a': 'D', 'b': 'D', 'c': 'D'}, 'D', id='single-frequency'),
            pytest.param({'a': 'D', 'b': 'M', 'c': 'D'}, 'D', id='daily-vs-monthly'),
            pytest.param({'a': 'h', 'b': 'D'}, 'h', id='hourly-vs-daily'),
            pytest.param({'a': 'W', 'b': 'M'}, 'W', id='weekly-vs-monthly'),
            # Fréquences positionnées et ancrées : l'ordre ne dépend que de la base
            pytest.param({'a': 'MS', 'b': 'QS-JAN'}, 'MS', id='positioned'),
            # Paires indétectables (None) et chaînes inconnues : ignorées
            pytest.param({'a': None, 'b': 'M', 'c': 'D'}, 'D', id='none-ignored'),
            pytest.param({'a': 'zzz', 'b': 'M'}, 'M', id='unknown-ignored'),
            pytest.param({'a': None, 'b': None}, None, id='only-none'),
            pytest.param({'a': 'zzz', 'b': 'yyy'}, None, id='only-unknown'),
        ],
    )
    def test_highest(self, frequency_map, expected):
        """Golden finest frequency of each map."""
        assert _get_highest_frequency(frequency_map) == expected
